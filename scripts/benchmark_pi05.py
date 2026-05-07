#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pi05_orin.paths import OUTPUTS_DIR, dataset_root, model_dir, split_dir
from pi05_orin.pi05_runtime import default_device, load_pi05_runtime
from pi05_orin.presets import PRESETS, get_preset
from pi05_orin.splits import read_episode_list
from pi05_orin.trt_runner import load_trt_pi05_runtime


PI05_PRESETS = [name for name, preset in PRESETS.items() if preset.model_family == "pi05"]


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pi0.5 inference latency on fixed eval episodes.")
    parser.add_argument("--preset", choices=sorted(PI05_PRESETS), default="pi05_aloha_public")
    parser.add_argument("--repo-id")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--model-ref")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=20)
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


def maybe_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


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
    iterator = iter(loader)

    with torch.no_grad():
        for _ in range(args.warmup_batches):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                break
            batch = runtime.preprocessor(raw_batch)
            runtime.policy.predict_action_chunk(batch)
            maybe_sync(args.device)

        preprocess_latencies_ms: list[float] = []
        policy_latencies_ms: list[float] = []
        end_to_end_latencies_ms: list[float] = []
        samples_measured = 0

        for _ in range(args.measure_batches):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                break

            maybe_sync(args.device)
            end_start = time.perf_counter()

            pre_start = time.perf_counter()
            batch = runtime.preprocessor(raw_batch)
            maybe_sync(args.device)
            preprocess_ms = (time.perf_counter() - pre_start) * 1000.0

            policy_start = time.perf_counter()
            pred = runtime.policy.predict_action_chunk(batch)
            maybe_sync(args.device)
            policy_ms = (time.perf_counter() - policy_start) * 1000.0
            end_ms = (time.perf_counter() - end_start) * 1000.0

            preprocess_latencies_ms.append(preprocess_ms)
            policy_latencies_ms.append(policy_ms)
            end_to_end_latencies_ms.append(end_ms)
            samples_measured += int(pred.shape[0])

    if not end_to_end_latencies_ms:
        raise RuntimeError("No benchmark batches were processed.")

    result = {
        "repo_id": repo_id,
        "model_ref": model_ref,
        "device": args.device,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "measure_batches": len(end_to_end_latencies_ms),
        "num_inference_steps": runtime.config.num_inference_steps,
        "samples_measured": samples_measured,
        "mean_preprocess_ms": statistics.fmean(preprocess_latencies_ms),
        "mean_policy_ms": statistics.fmean(policy_latencies_ms),
        "mean_end_to_end_ms": statistics.fmean(end_to_end_latencies_ms),
        "p50_end_to_end_ms": percentile(end_to_end_latencies_ms, 0.50),
        "p95_end_to_end_ms": percentile(end_to_end_latencies_ms, 0.95),
        "p50_policy_ms": percentile(policy_latencies_ms, 0.50),
        "p95_policy_ms": percentile(policy_latencies_ms, 0.95),
    }

    output = args.output or (OUTPUTS_DIR / "benchmarks" / f"{Path(model_ref).name}_pi05_latency.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
