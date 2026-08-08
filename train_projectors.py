"""Launch one independent StyCtrl projector training job per strength and GPU."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from collections import deque
from pathlib import Path

from train_lora import PROJECT_ROOT, build_finetune_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train independent projector anchors. Each available GPU runs one strength at a time; "
            "remaining strengths are queued."
        )
    )
    parser.add_argument("--model", choices=["qwen", "flux"], default="qwen")
    parser.add_argument("--pretrained-pipeline")
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument(
        "--projector-type",
        choices=["low_rank_scale", "low_rank_linear", "in_scale"],
        default="low_rank_linear",
    )
    parser.add_argument("--strengths", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--gpus", nargs="+", required=True, help="Physical GPU IDs, e.g. --gpus 0 1 2 3")

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
    parser.add_argument("--project-name-prefix", default="styctrl_projector")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-text-encoder", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--max-training-steps", type=int, default=5000)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=2500)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--train-num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    invalid_strengths = sorted(set(args.strengths) - set(range(11)))
    if invalid_strengths:
        raise SystemExit(f"Strength IDs must be within 0..10; got {invalid_strengths}.")
    if len(set(args.strengths)) != len(args.strengths):
        raise SystemExit("--strengths must not contain duplicates.")

    lora_checkpoint = Path(args.lora_checkpoint).expanduser().resolve()
    if not lora_checkpoint.is_file():
        raise SystemExit(f"LoRA checkpoint does not exist: {lora_checkpoint}")

    if args.model == "flux":
        load_text_encoder = True
    elif args.load_text_encoder is not None:
        load_text_encoder = args.load_text_encoder
    else:
        load_text_encoder = not (args.train_txt_latent_dir and args.val_txt_latent_dir)

    if args.model == "qwen" and not load_text_encoder:
        if not args.train_txt_latent_dir or not args.val_txt_latent_dir:
            raise SystemExit(
                "Qwen with --no-load-text-encoder requires both text-latent directories."
            )

    commands: dict[int, list[str]] = {}
    for strength in args.strengths:
        commands[strength] = build_finetune_command(
            model=args.model,
            pretrained_pipeline=args.pretrained_pipeline,
            dataset_dir=args.dataset_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            train_txt_latent_dir=args.train_txt_latent_dir,
            val_txt_latent_dir=args.val_txt_latent_dir,
            target_strengths=[strength],
            output_dir=args.output_dir,
            project_name=f"{args.project_name_prefix}_s{strength:02d}",
            rank=args.rank,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            proj_type=args.projector_type,
            checkpoint_path=str(lora_checkpoint),
            modules_require_grad=["projector"],
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

    if args.dry_run:
        for index, strength in enumerate(args.strengths):
            gpu = args.gpus[index % len(args.gpus)]
            print(f"[GPU {gpu}] strength={strength} w={strength / 10:.1f}")
            print(shlex.join(commands[strength]))
        return

    pending = deque(args.strengths)
    available_gpus = deque(str(gpu) for gpu in args.gpus)
    active: dict[subprocess.Popen, tuple[str, int]] = {}

    def launch(gpu: str, strength: int) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        print(f"[launch] GPU {gpu}: strength {strength} (w={strength / 10:.1f})", flush=True)
        process = subprocess.Popen(commands[strength], cwd=PROJECT_ROOT, env=env)
        active[process] = (gpu, strength)

    try:
        while pending or active:
            while pending and available_gpus:
                launch(available_gpus.popleft(), pending.popleft())

            finished = [process for process in active if process.poll() is not None]
            for process in finished:
                gpu, strength = active.pop(process)
                available_gpus.append(gpu)
                if process.returncode != 0:
                    for other in active:
                        other.terminate()
                    raise SystemExit(
                        f"Projector training failed on GPU {gpu}, strength {strength}, "
                        f"exit code {process.returncode}."
                    )
                print(f"[done] GPU {gpu}: strength {strength}", flush=True)

            if active and not finished:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted; terminating projector jobs...", flush=True)
        for process in active:
            process.terminate()
        for process in active:
            process.wait()
        raise


if __name__ == "__main__":
    main()
