"""Run one StyCtrl inference from prompt, content, style, and strength."""

from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from diffusers import FluxKontextPipeline
from PIL import Image
from safetensors.torch import load_file
from scipy.interpolate import BSpline, make_interp_spline

from inference import flux_styctrl_infer
from models.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from tuner import FluxKontextFinetuner, QwenImageEditFinetuner


DEFAULT_PRETRAINED_PIPELINES = {
    "qwen": "Qwen/Qwen-Image-Edit-2509",
    "flux": "black-forest-labs/FLUX.1-Kontext-dev",
}
DEFAULT_RANKS = {"qwen": 128, "flux": 32}
DEFAULT_LAYER_ENDS = {"qwen": 59, "flux": 57}
TARGET_MODULES = {
    "qwen": [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "img_mlp.net.2",
        "txt_mlp.net.2",
    ],
    "flux": [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.2",
        "ff_context.net.2",
    ],
}


def _projector_only(state: dict[str, torch.Tensor], path: Path) -> dict[str, torch.Tensor]:
    projector_state = {key: value for key, value in state.items() if ".projector." in key}
    if not projector_state:
        raise ValueError(f"No projector parameters found in {path}.")
    return projector_state


def _load_projector_checkpoints(paths: Sequence[str]) -> list[dict[str, torch.Tensor]]:
    states = []
    expected_keys: set[str] | None = None
    expected_shapes: dict[str, torch.Size] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Projector checkpoint does not exist: {path}")
        state = _projector_only(load_file(path, device="cpu"), path)
        keys = set(state)
        if expected_keys is None:
            expected_keys = keys
            expected_shapes = {key: value.shape for key, value in state.items()}
        elif keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(f"Projector keys do not match for {path}; missing={missing}, extra={extra}")
        else:
            mismatched = [key for key, value in state.items() if value.shape != expected_shapes[key]]
            if mismatched:
                raise ValueError(f"Projector tensor shapes do not match for {path}: {mismatched}")
        states.append(state)
    return states


def _zero_state_like(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value) for key, value in state.items()}


def _to_device_dtype(
    state: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=dtype) for key, value in state.items()}


def _open_clamped_knot_vector(
    strengths: Sequence[float], degree: int
) -> np.ndarray:
    """Build an open clamped knot vector from ordered strength parameters."""

    parameters = np.asarray(strengths, dtype=np.float64)
    interior = [
        parameters[index : index + degree].mean()
        for index in range(1, len(parameters) - degree)
    ]
    return np.concatenate(
        (
            np.repeat(parameters[0], degree + 1),
            np.asarray(interior, dtype=np.float64),
            np.repeat(parameters[-1], degree + 1),
        )
    )


def interpolate_projector_states(
    paths: Sequence[str],
    anchor_strengths: Sequence[float],
    query_strength: float,
    method: str,
    order: int,
    spline_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    endpoint_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Interpolate projector tensors at an explicit strength coordinate."""

    states = _load_projector_checkpoints(paths)
    if len(states) == 1:
        if np.isclose(query_strength, 0.0):
            return _to_device_dtype(_zero_state_like(states[0]), device, dtype)
        return _to_device_dtype(states[0], device, dtype)
    if len(anchor_strengths) != len(states):
        raise ValueError(
            "--anchor-strengths must contain exactly one value per projector checkpoint."
        )

    pairs = sorted(zip(anchor_strengths, states), key=lambda item: item[0])
    strengths = [float(item[0]) for item in pairs]
    states = [item[1] for item in pairs]
    if len(set(strengths)) != len(strengths):
        raise ValueError("Anchor strengths must be unique.")
    if strengths[0] < 0 or strengths[-1] > 1:
        raise ValueError("Anchor strengths must lie within [0, 1].")

    # A zero projector makes the StyCtrl LoRA contribution zero for all three
    # learned projector types. Use it as an explicit content-side endpoint.
    if strengths[0] > 0:
        strengths.insert(0, 0.0)
        states.insert(0, _zero_state_like(states[0]))

    if endpoint_state is not None and strengths[-1] < 1.0:
        if endpoint_state.keys() != states[0].keys():
            raise ValueError("Endpoint projector keys do not match anchor checkpoints.")
        mismatched = [
            key
            for key, value in endpoint_state.items()
            if value.shape != states[0][key].shape
        ]
        if mismatched:
            raise ValueError(
                f"Endpoint projector tensor shapes do not match anchors: {mismatched}"
            )
        strengths.append(1.0)
        states.append(endpoint_state)

    if query_strength < strengths[0] or query_strength > strengths[-1]:
        raise ValueError(
            f"Requested strength {query_strength} is outside the available projector range "
            f"[{strengths[0]}, {strengths[-1]}]."
        )

    if method != "bspline" or spline_mode == "interpolating":
        for anchor_strength, state in zip(strengths, states):
            if np.isclose(query_strength, anchor_strength):
                return _to_device_dtype(state, device, dtype)

    if method == "linear":
        upper = bisect.bisect_right(strengths, query_strength)
        lower = upper - 1
        x0, x1 = strengths[lower], strengths[upper]
        alpha = (query_strength - x0) / (x1 - x0)
        result = {
            key: (1.0 - alpha) * states[lower][key].float() + alpha * states[upper][key].float()
            for key in states[0]
        }
        return _to_device_dtype(result, device, dtype)

    if method != "bspline":
        raise ValueError(f"Unsupported interpolation method: {method}")
    if order < 1 or order >= len(strengths):
        raise ValueError(
            f"B-spline order must satisfy 1 <= order < number of anchors ({len(strengths)})."
        )

    x = np.asarray(strengths, dtype=np.float64)
    result = {}
    for key in states[0]:
        y = torch.stack([state[key].float() for state in states], dim=0).numpy()
        if spline_mode == "interpolating":
            value = make_interp_spline(x, y, k=order)(np.asarray(query_strength))
        elif spline_mode == "control":
            knots = _open_clamped_knot_vector(x, order)
            value = BSpline(knots, y, k=order, axis=0)(np.asarray(query_strength))
        else:
            raise ValueError(f"Unsupported spline mode: {spline_mode}")
        result[key] = torch.from_numpy(np.asarray(value))
    return _to_device_dtype(result, device, dtype)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one image from a prompt, content image, style image, and strength."
    )
    parser.add_argument("--model", choices=["qwen", "flux"], default="qwen")
    parser.add_argument("--pretrained-pipeline")
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument(
        "--projector-type",
        choices=["none", "lora", "low_rank_scale", "low_rank_linear", "in_scale"],
        default="low_rank_linear",
    )
    parser.add_argument("--projector-checkpoints", nargs="*", default=[])
    parser.add_argument("--anchor-strengths", nargs="*", type=float, default=[])
    parser.add_argument("--interpolation", choices=["linear", "bspline"], default="bspline")
    parser.add_argument("--bspline-order", type=int, default=3)
    parser.add_argument(
        "--spline-mode",
        choices=["control", "interpolating"],
        default="control",
    )

    parser.add_argument("--rank", type=int)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.strength <= 1:
        raise SystemExit("--strength must lie within [0, 1].")

    content_path = Path(args.content).expanduser().resolve()
    style_path = Path(args.style).expanduser().resolve()
    lora_checkpoint = Path(args.lora_checkpoint).expanduser().resolve()
    for label, path in [
        ("content image", content_path),
        ("style image", style_path),
        ("LoRA checkpoint", lora_checkpoint),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")

    if args.projector_type in {"low_rank_scale", "low_rank_linear", "in_scale"}:
        if not args.projector_checkpoints:
            raise SystemExit(
                f"projector type {args.projector_type} requires --projector-checkpoints."
            )
    elif args.projector_checkpoints:
        raise SystemExit(
            "--projector-checkpoints are only valid with low_rank_scale, "
            "low_rank_linear, or in_scale."
        )
    if len(args.projector_checkpoints) > 1 and not args.anchor_strengths:
        raise SystemExit("Multiple projector checkpoints require --anchor-strengths.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    pretrained_pipeline = args.pretrained_pipeline or DEFAULT_PRETRAINED_PIPELINES[args.model]
    rank = args.rank if args.rank is not None else DEFAULT_RANKS[args.model]
    layer_end = args.layer_end if args.layer_end is not None else DEFAULT_LAYER_ENDS[args.model]

    finetuner_cls = QwenImageEditFinetuner if args.model == "qwen" else FluxKontextFinetuner
    finetuner = finetuner_cls(
        pretrained_pipeline=pretrained_pipeline,
        lora_layer_range=[args.layer_start, layer_end],
        load_dit=True,
        load_text_encoder=True,
        torch_dtype=args.dtype,
        device=device,
    )
    finetuner.register_styctrl(
        lora_target_modules=TARGET_MODULES[args.model],
        device=device,
        dtype=torch_dtype,
    )
    finetuner.add_styctrl(
        rank=rank,
        adapter_name="styctrl",
        proj_type=args.projector_type,
        bias=False,
        checkpoint_path=str(lora_checkpoint),
        device=device,
        dtype=torch_dtype,
    )
    finetuner.activate_styctrl("styctrl")

    if args.projector_checkpoints:
        endpoint_projector_state = {
            key: value.detach().cpu()
            for key, value in finetuner.transformer.state_dict().items()
            if ".projector." in key
        }
        projector_state = interpolate_projector_states(
            paths=args.projector_checkpoints,
            anchor_strengths=args.anchor_strengths,
            query_strength=args.strength,
            method=args.interpolation,
            order=args.bspline_order,
            spline_mode=args.spline_mode,
            device=device,
            dtype=torch_dtype,
            endpoint_state=endpoint_projector_state,
        )
        finetuner.transformer.load_state_dict(projector_state, strict=False)
    elif args.projector_type == "none" and not np.isclose(args.strength, 1.0):
        print(
            "warning: projector-type=none ignores --strength; the full LoRA will be used.",
            file=sys.stderr,
        )

    content_image = Image.open(content_path).convert("RGB").resize((args.width, args.height))
    style_image = Image.open(style_path).convert("RGB").resize((args.width, args.height))
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    strength = torch.tensor([args.strength], device=device, dtype=torch_dtype)
    attention_kwargs = {
        "lora_layer_indices": finetuner.lora_layer_indices,
        "enable_lora": True,
        "w": strength,
    }

    if args.model == "flux":
        pipeline = FluxKontextPipeline(
            transformer=finetuner.transformer,
            vae=finetuner.vae,
            scheduler=finetuner.scheduler_val,
            text_encoder=finetuner.text_encoder,
            tokenizer=finetuner.tokenizer,
            text_encoder_2=finetuner.text_encoder_2,
            tokenizer_2=finetuner.tokenizer_2,
        )
        output_image = flux_styctrl_infer(
            pipeline=pipeline,
            content_image=content_image,
            style_image=style_image,
            prompt=args.prompt,
            joint_attention_kwargs=attention_kwargs,
            num_inference_steps=args.steps,
            generator=generator,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        )
    else:
        pipeline = QwenImageEditPlusPipeline(
            transformer=finetuner.transformer,
            vae=finetuner.vae,
            scheduler=finetuner.scheduler_val,
            text_encoder=finetuner.text_encoder,
            tokenizer=finetuner.tokenizer,
            processor=finetuner.processor,
        )
        output_image = pipeline(
            image=[content_image, style_image],
            prompt=args.prompt,
            attention_kwargs=attention_kwargs,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

    output_image.save(output_path)
    print(f"Saved strength {args.strength:.3f} result to {output_path}")


if __name__ == "__main__":
    main()
