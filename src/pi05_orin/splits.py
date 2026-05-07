from __future__ import annotations

import json
import random
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_episode_indices(episodes_obj) -> list[int]:
    if hasattr(episodes_obj, "column_names"):
        columns = set(getattr(episodes_obj, "column_names"))
        for name in ("episode_index", "episode_idx", "index"):
            if name in columns:
                return sorted(int(v) for v in episodes_obj[name])
        return list(range(len(episodes_obj)))
    if hasattr(episodes_obj, "columns"):
        columns = set(getattr(episodes_obj, "columns"))
        for name in ("episode_index", "episode_idx", "index"):
            if name in columns:
                return sorted(int(v) for v in episodes_obj[name].tolist())
        if getattr(episodes_obj.index, "dtype", None) is not None:
            return sorted(int(v) for v in episodes_obj.index.tolist())
        return list(range(len(episodes_obj)))
    if isinstance(episodes_obj, list):
        return [int(v) for v in episodes_obj]
    raise TypeError(f"Unsupported episodes object type: {type(episodes_obj)!r}")


def build_fixed_split(
    episode_indices: list[int],
    eval_episodes: int,
    train_max_episodes: int | None,
    strategy: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be > 0")
    if len(episode_indices) <= eval_episodes:
        raise ValueError(
            f"Dataset has only {len(episode_indices)} episodes, cannot reserve {eval_episodes} for eval."
        )

    ordered = sorted(int(v) for v in episode_indices)
    if strategy == "first":
        eval_ids = ordered[:eval_episodes]
    elif strategy == "random":
        rng = random.Random(seed)
        eval_ids = sorted(rng.sample(ordered, k=eval_episodes))
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    eval_set = set(eval_ids)
    train_ids = [ep for ep in ordered if ep not in eval_set]
    if train_max_episodes is not None:
        train_ids = train_ids[:train_max_episodes]

    if not train_ids:
        raise ValueError("Train split is empty after applying train_max_episodes.")

    return train_ids, eval_ids


def write_split(output_dir: Path, repo_id: str, train_ids: list[int], eval_ids: list[int], metadata: dict) -> None:
    ensure_dir(output_dir)
    (output_dir / "train_episodes.json").write_text(json.dumps(train_ids, indent=2))
    (output_dir / "eval_episodes.json").write_text(json.dumps(eval_ids, indent=2))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "train_episodes": len(train_ids),
                "eval_episodes": len(eval_ids),
                **metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def read_episode_list(path: Path) -> list[int]:
    return [int(v) for v in json.loads(path.read_text())]
