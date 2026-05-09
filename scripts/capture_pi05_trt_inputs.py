#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from pi05_orin.paths import (
    dataset_root,
    model_dir,
    split_dir,
    trt_capture_dir,
)
from pi05_orin.pi05_runtime import default_device, load_pi05_runtime
from pi05_orin.presets import PRESETS, get_preset
from pi05_orin.splits import read_episode_list
from pi05_orin.trt_wrappers import (
    Pi05PrefixEmbedWrapper,
    Pi05PrefixLmWrapper,
    Pi05SuffixStepWrapper,
    prefix_cache_names,
)


PI05_PRESETS = [name for name, preset in PRESETS.items() if preset.model_family == "pi05"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture representative tensors for PI0.5 TRT export/calibration.")
    parser.add_argument("--preset", choices=sorted(PI05_PRESETS), default="pi05_aloha_public")
    parser.add_argument("--repo-id")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--model-ref")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-samples-per-episode", type=int)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--model-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--capture-suffix-step", action="store_true")
    parser.add_argument("--capture-suffix-loop", action="store_true")
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


def select_episodes(split_path: Path, split_name: str) -> list[int]:
    filename = "train_episodes.json" if split_name == "train" else "eval_episodes.json"
    return read_episode_list(split_path / filename)


def scalar_to_int(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def main() -> None:
    args = parse_args()
    if args.tokenizer_name:
        os.environ["PI05_TOKENIZER_NAME"] = args.tokenizer_name

    torch.manual_seed(args.seed)
    preset = get_preset(args.preset)
    repo_id = args.repo_id or preset.repo_id
    root = args.root or dataset_root(repo_id)
    split_path = args.split_dir or split_dir(repo_id)
    episodes = select_episodes(split_path, args.split)
    allowed_episodes = set(episodes)
    model_ref = resolve_model_ref(args.preset, args.model_ref)
    output_root = args.output_dir or trt_capture_dir(args.preset)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime = load_pi05_runtime(
        model_ref=model_ref,
        repo_id=repo_id,
        root=root,
        episodes=None,
        device=args.device,
        revision=args.revision,
        dtype_override=args.model_dtype,
    )
    prefix_embed = Pi05PrefixEmbedWrapper(runtime.policy).eval()
    prefix_lm = Pi05PrefixLmWrapper(runtime.policy).eval()
    suffix_step = Pi05SuffixStepWrapper(runtime.policy).eval()
    cache_names = prefix_cache_names(prefix_lm.num_layers)
    image_key = prefix_embed.image_key

    captured = 0
    captured_per_episode: dict[int, int] = {}
    for dataset_idx in range(args.sample_offset, len(runtime.dataset)):
        if captured >= args.num_samples:
            break

        raw_item = runtime.dataset[dataset_idx]
        episode_index = scalar_to_int(raw_item["episode_index"])
        if episode_index not in allowed_episodes:
            continue
        if args.max_samples_per_episode is not None:
            if captured_per_episode.get(episode_index, 0) >= args.max_samples_per_episode:
                continue

        raw_batch = default_collate([raw_item])
        batch = runtime.preprocessor(raw_batch)
        image = batch[image_key]
        tokens = batch["observation.language.tokens"]
        attention_mask = batch["observation.language.attention_mask"]

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_position_ids, prefix_attention_mask_4d = prefix_embed(
                image,
                tokens,
                attention_mask,
            )
            flat_cache = list(prefix_lm(prefix_embs, prefix_position_ids, prefix_attention_mask_4d))

            sample_dir = output_root / f"{args.split}_sample_{dataset_idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            inputs_payload = {
                "image": image.detach().cpu(),
                "tokens": tokens.detach().cpu(),
                "attention_mask": attention_mask.detach().cpu(),
                "target_action": batch["action"].detach().cpu(),
            }
            torch.save(inputs_payload, sample_dir / "inputs.pt")

            prefix_payload: dict[str, torch.Tensor] = {
                "prefix_embs": prefix_embs.detach().cpu(),
                "prefix_pad_masks": prefix_pad_masks.detach().cpu(),
                "prefix_position_ids": prefix_position_ids.detach().cpu(),
                "prefix_attention_mask_4d": prefix_attention_mask_4d.detach().cpu(),
            }

            summary = {
                "preset": args.preset,
                "split": args.split,
                "sample_index": dataset_idx,
                "image_key": image_key,
                "cache_tensors": cache_names,
                "inputs": {name: list(tensor.shape) for name, tensor in inputs_payload.items()},
                "prefix": {name: list(tensor.shape) for name, tensor in prefix_payload.items()},
            }

            if args.capture_suffix_step:
                dt = -1.0 / runtime.config.num_inference_steps
                dt_tensor = torch.tensor([dt], device=image.device, dtype=torch.float32)
                noise = runtime.policy.model.sample_noise(
                    (
                        image.shape[0],
                        runtime.config.chunk_size,
                        runtime.config.max_action_dim,
                    ),
                    device=image.device,
                )
                timestep = torch.ones((image.shape[0],), device=image.device, dtype=torch.float32)
                suffix_out = suffix_step(prefix_pad_masks, noise, timestep, dt_tensor, *flat_cache)
                suffix_payload = {
                    "prefix_pad_masks": prefix_pad_masks.detach().cpu(),
                    "x_t": noise.detach().cpu(),
                    "timestep": timestep.detach().cpu(),
                    "dt": dt_tensor.detach().cpu(),
                    "x_t_next": suffix_out.detach().cpu(),
                }
                torch.save(suffix_payload, sample_dir / "suffix_step.pt")
                summary["suffix_step"] = {name: list(tensor.shape) for name, tensor in suffix_payload.items()}

            if args.capture_suffix_loop:
                x_t = runtime.policy.model.sample_noise(
                    (
                        image.shape[0],
                        runtime.config.chunk_size,
                        runtime.config.max_action_dim,
                    ),
                    device=image.device,
                )
                dt = -1.0 / runtime.config.num_inference_steps
                dt_tensor = torch.tensor([dt], device=image.device, dtype=torch.float32)
                suffix_loop_steps: list[dict[str, torch.Tensor | int | float]] = []
                for step_idx in range(runtime.config.num_inference_steps):
                    time_value = 1.0 + step_idx * dt
                    timestep = torch.full((image.shape[0],), time_value, device=image.device, dtype=torch.float32)
                    suffix_loop_steps.append(
                        {
                            "step_index": step_idx,
                            "timestep": timestep.detach().cpu(),
                            "x_t": x_t.detach().cpu(),
                        }
                    )
                    x_t = suffix_step(prefix_pad_masks, x_t, timestep, dt_tensor, *flat_cache)

                torch.save(
                    {
                        "dt": dt,
                        "num_inference_steps": runtime.config.num_inference_steps,
                        "steps": suffix_loop_steps,
                    },
                    sample_dir / "suffix_loop.pt",
                )
                summary["suffix_loop"] = {
                    "num_steps": runtime.config.num_inference_steps,
                    "x_t_shape": list(suffix_loop_steps[0]["x_t"].shape) if suffix_loop_steps else None,
                }

            del prefix_embs, prefix_position_ids, prefix_attention_mask_4d
            for idx, (name, tensor) in enumerate(zip(cache_names, flat_cache, strict=True)):
                prefix_payload[name] = tensor.detach().cpu()
                flat_cache[idx] = None
            del flat_cache

            torch.save(prefix_payload, sample_dir / "prefix.pt")
            del prefix_payload, prefix_pad_masks
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

            (sample_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            captured += 1
            captured_per_episode[episode_index] = captured_per_episode.get(episode_index, 0) + 1

    if captured == 0:
        raise RuntimeError("No samples were captured.")


if __name__ == "__main__":
    main()
