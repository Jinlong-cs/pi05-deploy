#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pi05_orin.paths import OUTPUTS_DIR, dataset_root, model_dir, split_dir
from pi05_orin.pi05_runtime import default_device, load_pi05_runtime
from pi05_orin.presets import PRESETS, get_preset
from pi05_orin.splits import read_episode_list
from pi05_orin.trt_runner import load_trt_pi05_runtime


PI05_PRESETS = [name for name, preset in PRESETS.items() if preset.model_family == "pi05"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed open-loop evaluation for a pi0.5 checkpoint.")
    parser.add_argument("--preset", choices=sorted(PI05_PRESETS), default="pi05_aloha_public")
    parser.add_argument("--repo-id")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--model-ref")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--backend", choices=["pytorch", "trt_fp16", "trt_int8"], default="pytorch")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def resolve_model_ref(preset_name: str, explicit_ref: str | None) -> str:
    if explicit_ref is not None:
        return explicit_ref
    preset = get_preset(preset_name)
    if preset.model_repo_id is None:
        raise ValueError("This preset does not define a default model repo id.")
    local_dir = model_dir(preset.model_repo_id)
    if local_dir.exists():
        return str(local_dir)
    return preset.model_repo_id


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    repo_id = args.repo_id or preset.repo_id
    root = args.root or dataset_root(repo_id)
    split_path = args.split_dir or split_dir(repo_id)
    eval_episodes = read_episode_list(split_path / "eval_episodes.json")
    model_ref = resolve_model_ref(args.preset, args.model_ref)

    if args.backend == "pytorch":
        runtime = load_pi05_runtime(
            model_ref=model_ref,
            repo_id=repo_id,
            root=root,
            episodes=eval_episodes,
            device=args.device,
            revision=args.revision,
            compile_model=args.compile_model,
            num_inference_steps=args.num_inference_steps,
        )
    else:
        runtime = load_trt_pi05_runtime(
            model_ref=model_ref,
            repo_id=repo_id,
            root=root,
            episodes=eval_episodes,
            device=args.device,
            revision=args.revision,
            preset_name=args.preset,
            precision="fp16" if args.backend == "trt_fp16" else "int8",
            num_inference_steps=args.num_inference_steps,
        )

    loader = DataLoader(
        runtime.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = {
        "num_batches": 0,
        "num_samples": 0,
        "normalized_action_mse_sum": 0.0,
        "normalized_action_l1_sum": 0.0,
        "normalized_first_action_mse_sum": 0.0,
        "normalized_first_action_l1_sum": 0.0,
    }

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if args.limit_batches is not None and batch_idx >= args.limit_batches:
                break

            batch = runtime.preprocessor(raw_batch)
            pred = runtime.policy.predict_action_chunk(batch)
            target = batch["action"]

            batch_size = int(target.shape[0])
            metrics["num_batches"] += 1
            metrics["num_samples"] += batch_size
            metrics["normalized_action_mse_sum"] += float(torch.mean((pred - target) ** 2).item())
            metrics["normalized_action_l1_sum"] += float(torch.mean(torch.abs(pred - target)).item())
            metrics["normalized_first_action_mse_sum"] += float(torch.mean((pred[:, 0] - target[:, 0]) ** 2).item())
            metrics["normalized_first_action_l1_sum"] += float(
                torch.mean(torch.abs(pred[:, 0] - target[:, 0])).item()
            )

    if metrics["num_batches"] == 0:
        raise RuntimeError("No evaluation batches were processed.")

    result = {
        "repo_id": repo_id,
        "model_ref": model_ref,
        "device": args.device,
        "backend": args.backend,
        "num_eval_episodes": len(eval_episodes),
        "num_inference_steps": runtime.config.num_inference_steps,
        "num_batches": metrics["num_batches"],
        "num_samples": metrics["num_samples"],
        "avg_normalized_action_mse": metrics["normalized_action_mse_sum"] / metrics["num_batches"],
        "avg_normalized_action_l1": metrics["normalized_action_l1_sum"] / metrics["num_batches"],
        "avg_normalized_first_action_mse": metrics["normalized_first_action_mse_sum"] / metrics["num_batches"],
        "avg_normalized_first_action_l1": metrics["normalized_first_action_l1_sum"] / metrics["num_batches"],
    }

    output = args.output or (OUTPUTS_DIR / "eval" / f"{Path(model_ref).name}_pi05_open_loop.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
