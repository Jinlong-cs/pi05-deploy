#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from pi05_orin.paths import model_dir
from pi05_orin.presets import DEFAULT_PRESET, PRESETS, get_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pi0.5 weights and config files locally.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--model-repo-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--allow-pattern", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    repo_id = args.model_repo_id or preset.model_repo_id
    if repo_id is None:
        raise ValueError("This preset does not define a default model repo id.")

    output_dir = args.output_dir or model_dir(repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = args.allow_pattern or [
        "*.json",
        "*.safetensors",
        "*.md",
        "*.txt",
        "*.model",
    ]

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=output_dir,
        allow_patterns=allow_patterns,
    )

    result = {
        "model_repo_id": repo_id,
        "revision": args.revision,
        "local_dir": str(output_dir),
        "snapshot_path": str(snapshot_path),
        "allow_patterns": allow_patterns,
    }
    (output_dir / "download_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
