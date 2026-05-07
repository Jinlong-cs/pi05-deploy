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
from pi05_orin.trt_wrappers import (
    Pi05PrefixEmbedWrapper,
    Pi05PrefixLmWrapper,
    prefix_cache_names,
    save_io_summary,
)


PI05_PRESETS = [name for name, preset in PRESETS.items() if preset.model_family == "pi05"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PI0.5 prefix wrappers to ONNX.")
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
    parser.add_argument("--stage", choices=["all", "prefix_embed", "prefix_lm"], default="all")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-dir", type=Path)
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
    try:
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"warning: onnx validation skipped for {path}: {exc}")


def ensure_onnx_available() -> None:
    try:
        import onnx  # noqa: F401
        return
    except Exception:
        pass

    import sys

    for candidate in ("/usr/lib/python3/dist-packages", "/usr/lib/python3.10/dist-packages"):
        if candidate not in sys.path and Path(candidate).exists():
            sys.path.append(candidate)
            try:
                import onnx  # noqa: F401

                return
            except Exception:
                continue

    raise RuntimeError("The active Python environment is missing the 'onnx' package required by torch.onnx.export.")


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
    prefix_embed = Pi05PrefixEmbedWrapper(runtime.policy).eval()
    prefix_lm = Pi05PrefixLmWrapper(runtime.policy).eval()

    inputs_payload = torch.load(capture_dir / "inputs.pt", map_location=args.device)
    prefix_payload = torch.load(capture_dir / "prefix.pt", map_location=args.device)

    image = inputs_payload["image"].to(device=args.device)
    tokens = inputs_payload["tokens"].to(device=args.device)
    attention_mask = inputs_payload["attention_mask"].to(device=args.device)

    exported_paths: list[Path] = []
    prefix_embed_inputs = (image, tokens, attention_mask)
    if args.stage in {"all", "prefix_embed"}:
        prefix_embed_path = output_dir / "prefix_embed.onnx"
        torch.onnx.export(
            prefix_embed,
            prefix_embed_inputs,
            str(prefix_embed_path),
            input_names=["image", "tokens", "attention_mask"],
            output_names=["prefix_embs", "prefix_pad_masks", "prefix_position_ids", "prefix_attention_mask_4d"],
            opset_version=args.opset,
        )
        with torch.no_grad():
            embed_outputs = prefix_embed(*prefix_embed_inputs)
        save_io_summary(
            output_dir / "prefix_embed_io.json",
            inputs={
                "image": image,
                "tokens": tokens,
                "attention_mask": attention_mask,
            },
            outputs={
                "prefix_embs": embed_outputs[0],
                "prefix_pad_masks": embed_outputs[1],
                "prefix_position_ids": embed_outputs[2],
                "prefix_attention_mask_4d": embed_outputs[3],
            },
        )
        maybe_check_onnx(prefix_embed_path)
        exported_paths.append(prefix_embed_path)

    if args.stage in {"all", "prefix_lm"}:
        prefix_embs = prefix_payload["prefix_embs"].to(device=args.device)
        prefix_position_ids = prefix_payload["prefix_position_ids"].to(device=args.device)
        prefix_attention_mask_4d = prefix_payload["prefix_attention_mask_4d"].to(device=args.device)
        prefix_lm_path = output_dir / "prefix_lm.onnx"
        prefix_lm_inputs = (prefix_embs, prefix_position_ids, prefix_attention_mask_4d)
        prefix_lm_output_names = prefix_cache_names(prefix_lm.num_layers)
        torch.onnx.export(
            prefix_lm,
            prefix_lm_inputs,
            str(prefix_lm_path),
            input_names=["prefix_embs", "prefix_position_ids", "prefix_attention_mask_4d"],
            output_names=prefix_lm_output_names,
            opset_version=args.opset,
        )
        with torch.no_grad():
            prefix_lm_outputs = prefix_lm(*prefix_lm_inputs)
        save_io_summary(
            output_dir / "prefix_lm_io.json",
            inputs={
                "prefix_embs": prefix_embs,
                "prefix_position_ids": prefix_position_ids,
                "prefix_attention_mask_4d": prefix_attention_mask_4d,
            },
            outputs={
                name: tensor for name, tensor in zip(prefix_lm_output_names, prefix_lm_outputs, strict=True)
            },
        )
        maybe_check_onnx(prefix_lm_path)
        exported_paths.append(prefix_lm_path)

    for path in exported_paths:
        print(path)


if __name__ == "__main__":
    main()
