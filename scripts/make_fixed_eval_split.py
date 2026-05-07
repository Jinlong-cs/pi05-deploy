#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from pi05_orin.paths import dataset_root, split_dir
from pi05_orin.presets import DEFAULT_PRESET, PRESETS, get_preset
from pi05_orin.splits import build_fixed_split, get_episode_indices, write_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed train/eval episode split for pi0.5.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--repo-id")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--train-max-episodes", type=int)
    parser.add_argument("--strategy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    repo_id = args.repo_id or preset.repo_id
    root = args.root or dataset_root(repo_id)
    output_dir = args.output_dir or split_dir(repo_id)
    eval_episodes = args.eval_episodes or preset.eval_episodes
    train_max_episodes = args.train_max_episodes or preset.train_max_episodes

    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=args.revision)
    episode_indices = get_episode_indices(meta.episodes)
    train_ids, eval_ids = build_fixed_split(
        episode_indices=episode_indices,
        eval_episodes=eval_episodes,
        train_max_episodes=train_max_episodes,
        strategy=args.strategy,
        seed=args.seed,
    )

    write_split(
        output_dir=output_dir,
        repo_id=repo_id,
        train_ids=train_ids,
        eval_ids=eval_ids,
        metadata={
            "seed": args.seed,
            "strategy": args.strategy,
            "root": str(root),
        },
    )

    print(
        json.dumps(
            {
                "repo_id": repo_id,
                "root": str(root),
                "output_dir": str(output_dir),
                "train_episodes": len(train_ids),
                "eval_episodes": len(eval_ids),
                "train_preview": train_ids[:10],
                "eval_preview": eval_ids[:10],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
