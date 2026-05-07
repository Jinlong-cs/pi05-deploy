#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pi05_orin.paths import dataset_root
from pi05_orin.presets import DEFAULT_PRESET, PRESETS, get_preset
from pi05_orin.splits import get_episode_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the public dataset used by the PI0.5 Orin deploy flow.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--repo-id")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--mode", choices=["metadata", "full"], default="metadata")
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--force-cache-sync", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def run_with_retries(fn, retries: int):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(attempt * 2, 8))
    raise last_error


def summarize(meta: LeRobotDatasetMetadata, root: Path) -> dict:
    feature_keys = sorted(meta.features.keys())
    camera_keys = [
        key for key in feature_keys if key == "observation.image" or key.startswith("observation.images")
    ]
    episode_indices = get_episode_indices(meta.episodes)
    summary = {
        "repo_id": meta.repo_id,
        "root": str(root),
        "revision": meta.revision,
        "num_episodes": len(episode_indices),
        "episode_range": [min(episode_indices), max(episode_indices)] if episode_indices else [],
        "fps": meta.info.get("fps"),
        "feature_keys": feature_keys,
        "camera_keys": camera_keys,
    }
    (root / "download_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    repo_id = args.repo_id or preset.repo_id
    root = args.root or dataset_root(repo_id)
    root.mkdir(parents=True, exist_ok=True)
    download_videos = args.download_videos or args.mode == "full"

    if args.mode == "metadata":
        meta = run_with_retries(
            lambda: LeRobotDatasetMetadata(
                repo_id,
                root=root,
                revision=args.revision,
                force_cache_sync=args.force_cache_sync,
            ),
            args.retries,
        )
    else:
        dataset = run_with_retries(
            lambda: LeRobotDataset(
                repo_id,
                root=root,
                revision=args.revision,
                force_cache_sync=args.force_cache_sync,
                download_videos=download_videos,
                video_backend="pyav",
            ),
            args.retries,
        )
        meta = dataset.meta

    summary = summarize(meta, root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
