"""Train the base StyCtrl LoRA on the strongest synthesized style level."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
FINETUNE_ENTRYPOINT = PROJECT_ROOT / "finetune.py"
STRONGEST_STRENGTH_ID = 10

DEFAULT_PRETRAINED_PIPELINES = {
    "qwen": "Qwen/Qwen-Image-Edit-2509",
    "flux": "black-forest-labs/FLUX.1-Kontext-dev",
}
DEFAULT_RANKS = {"qwen": 128, "flux": 32}
DEFAULT_LAYER_ENDS = {"qwen": 59, "flux": 57}


def _absolute_path(path: str | None) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve())


def _hydra_optional_path(path: str | None) -> str:
    return _absolute_path(path) if path else "null"


def _hydra_list(values: Sequence[object]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def build_finetune_command(
    *,
    model: str,
    pretrained_pipeline: str | None,
    dataset_dir: str,
    train_split: str,
    val_split: str,
    train_txt_latent_dir: str | None,
    val_txt_latent_dir: str | None,
    target_strengths: Sequence[int],
    output_dir: str,
    project_name: str,
    rank: int | None,
    layer_start: int,
    layer_end: int | None,
    proj_type: str,
    checkpoint_path: str | None,
    modules_require_grad: Sequence[str],
    bias: bool,
    load_text_encoder: bool,
    max_training_steps: int,
    checkpointing_steps: int,
    validation_steps: int,
    gradient_accumulation: int,
    train_num_workers: int,
    val_num_workers: int,
    seed: int,
    mixed_precision: str,
) -> list[str]:
    """Build a single-process Hydra training command used by both launchers."""

    config_name = "default" if model == "qwen" else "flux_kontext"
    pretrained_pipeline = pretrained_pipeline or DEFAULT_PRETRAINED_PIPELINES[model]
    rank = rank if rank is not None else DEFAULT_RANKS[model]
    layer_end = layer_end if layer_end is not None else DEFAULT_LAYER_ENDS[model]

    overrides = [
        f"output_dir={_absolute_path(output_dir)}",
        f"project_name={project_name}",
        f"seed={seed}",
        f"mixed_precision={mixed_precision}",
        "train_batch_size=1",
        "val_batch_size=1",
        f"gradient_accumulation={gradient_accumulation}",
        f"train_num_workers={train_num_workers}",
        f"val_num_workers={val_num_workers}",
        f"max_training_steps={max_training_steps}",
        f"checkpointing_steps={checkpointing_steps}",
        f"validation_steps={validation_steps}",
        f"trainset.dataset_dir={_absolute_path(dataset_dir)}",
        f"trainset.split={train_split}",
        f"trainset.txt_latent_dir={_hydra_optional_path(train_txt_latent_dir)}",
        f"trainset.target_strength={_hydra_list(target_strengths)}",
        f"valset.dataset_dir={_absolute_path(dataset_dir)}",
        f"valset.split={val_split}",
        f"valset.txt_latent_dir={_hydra_optional_path(val_txt_latent_dir)}",
        f"valset.target_strength={_hydra_list(target_strengths)}",
        f"finetuner.pretrained_pipeline={pretrained_pipeline}",
        f"finetuner.load_text_encoder={str(load_text_encoder).lower()}",
        f"finetuner.modules_require_grad={_hydra_list(modules_require_grad)}",
        f"styctrl.lora_layer_range={_hydra_list([layer_start, layer_end])}",
        f"styctrl.rank={rank}",
        f"styctrl.proj_type={proj_type}",
        f"styctrl.bias={str(bias).lower()}",
        f"styctrl.checkpoint_path={_absolute_path(checkpoint_path) if checkpoint_path else 'null'}",
    ]
    return [
        sys.executable,
        str(FINETUNE_ENTRYPOINT),
        "--config-name",
        config_name,
        *overrides,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train StyCtrl LoRA A/B using only synthesized strength ID 10 (w=1.0)."
    )
    parser.add_argument("--model", choices=["qwen", "flux"], default="qwen")
    parser.add_argument("--pretrained-pipeline")

    parser.add_argument(
        "--dataset-dir",
        default="SmoothStyle",
        help="Root of the local SmoothStyle repository checkout.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--train-txt-latent-dir")
    parser.add_argument("--val-txt-latent-dir")

    parser.add_argument("--output-dir", default="experiment/outputs")
    parser.add_argument("--project-name", default="styctrl_lora_s10")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-text-encoder", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--max-training-steps", type=int, default=5000)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--train-num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gpu", help="Physical GPU ID exposed to this process, for example 0.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.model == "flux":
        load_text_encoder = True
    elif args.load_text_encoder is not None:
        load_text_encoder = args.load_text_encoder
    else:
        load_text_encoder = not (args.train_txt_latent_dir and args.val_txt_latent_dir)

    if args.model == "qwen" and not load_text_encoder:
        if not args.train_txt_latent_dir or not args.val_txt_latent_dir:
            raise SystemExit(
                "Qwen with --no-load-text-encoder requires both --train-txt-latent-dir "
                "and --val-txt-latent-dir."
            )

    command = build_finetune_command(
        model=args.model,
        pretrained_pipeline=args.pretrained_pipeline,
        dataset_dir=args.dataset_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        train_txt_latent_dir=args.train_txt_latent_dir,
        val_txt_latent_dir=args.val_txt_latent_dir,
        target_strengths=[STRONGEST_STRENGTH_ID],
        output_dir=args.output_dir,
        project_name=args.project_name,
        rank=args.rank,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        proj_type="none",
        checkpoint_path=None,
        modules_require_grad=["lora_A", "lora_B"],
        bias=args.bias,
        load_text_encoder=load_text_encoder,
        max_training_steps=args.max_training_steps,
        checkpointing_steps=args.checkpointing_steps,
        validation_steps=args.validation_steps,
        gradient_accumulation=args.gradient_accumulation,
        train_num_workers=args.train_num_workers,
        val_num_workers=args.val_num_workers,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )

    print(shlex.join(command), flush=True)
    if args.dry_run:
        return

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
