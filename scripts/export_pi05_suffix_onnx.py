#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from pi05_orin.paths import dataset_root, model_dir, split_dir, trt_capture_dir, trt_onnx_dir
from pi05_orin.pi05_runtime import default_device, load_pi05_runtime
from pi05_orin.presets import PRESETS, get_preset
from pi05_orin.splits import read_episode_list
from pi05_orin.trt_wrappers import Pi05SuffixStepWrapper, prefix_cache_names, save_io_summary


PI05_PRESETS = [name for name, preset in PRESETS.items() if preset.model_family == "pi05"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PI0.5 suffix denoise step wrapper to ONNX.")
    parser.add_argument("--preset", choices=sorted(PI05_PRESETS), default="pi05_aloha_public")
    parser.add_argument("--repo-id")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--model-ref")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--model-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
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


def maybe_check_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def ensure_onnx_available() -> None:
    import onnx  # noqa: F401


def main() -> None:
    args = parse_args()
    ensure_onnx_available()
    if args.tokenizer_name:
        os.environ["PI05_TOKENIZER_NAME"] = args.tokenizer_name

    preset = get_preset(args.preset)
    repo_id = args.repo_id or preset.repo_id
    root = args.root or dataset_root(repo_id)
    split_path = args.split_dir or split_dir(repo_id)
    model_ref = resolve_model_ref(args.preset, args.model_ref)
    output_dir = args.output_dir or trt_onnx_dir(args.preset)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_dir = args.capture_dir
    if capture_dir is None:
        candidates = sorted(trt_capture_dir(args.preset).glob("*_sample_*"))
        if not candidates:
            raise FileNotFoundError("No capture directory found. Run capture_pi05_trt_inputs.py first.")
        capture_dir = candidates[0]

    episodes = read_episode_list(split_path / "eval_episodes.json")
    runtime = load_pi05_runtime(
        model_ref=model_ref,
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        device=args.device,
        revision=args.revision,
        dtype_override=args.model_dtype,
    )
    suffix_step = Pi05SuffixStepWrapper(runtime.policy).eval()
    cache_names = prefix_cache_names(suffix_step.num_layers)

    prefix_payload = torch.load(capture_dir / "prefix.pt", map_location=args.device)

    prefix_pad_masks = prefix_payload["prefix_pad_masks"].to(device=args.device)
    torch.manual_seed(args.seed)
    x_t = torch.randn(
        (
            prefix_pad_masks.shape[0],
            runtime.config.chunk_size,
            runtime.config.max_action_dim,
        ),
        dtype=torch.float32,
        device=args.device,
    )
    timestep = torch.ones((prefix_pad_masks.shape[0],), dtype=torch.float32, device=args.device)
    dt = torch.tensor([-1.0 / runtime.config.num_inference_steps], dtype=torch.float32, device=args.device)
    cache_tensors = [prefix_payload[name].to(device=args.device) for name in cache_names]

    suffix_path = output_dir / "suffix_step.onnx"
    input_names = ["prefix_pad_masks", "x_t", "timestep", "dt", *cache_names]
    torch.onnx.export(
        suffix_step,
        (prefix_pad_masks, x_t, timestep, dt, *cache_tensors),
        str(suffix_path),
        input_names=input_names,
        output_names=["x_t_next"],
        opset_version=args.opset,
    )
    with torch.no_grad():
        x_t_next = suffix_step(prefix_pad_masks, x_t, timestep, dt, *cache_tensors)
    save_io_summary(
        output_dir / "suffix_step_io.json",
        inputs={
            "prefix_pad_masks": prefix_pad_masks,
            "x_t": x_t,
            "timestep": timestep,
            "dt": dt,
            **{name: tensor for name, tensor in zip(cache_names, cache_tensors, strict=True)},
        },
        outputs={"x_t_next": x_t_next},
    )
    maybe_check_onnx(suffix_path)

    print(suffix_path)


if __name__ == "__main__":
    main()
