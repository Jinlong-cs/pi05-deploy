#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "pi05_minimal_fixture.json"
REQUIRED_RUNTIME_CONTRACT = {
    "tokenizer_max_length": 200,
    "prefix_length": 456,
    "action_latent_shape": [1, 50, 32],
    "action_chunk_shape": [1, 50, 14],
    "num_inference_steps": 10,
    "trt_stages": ["prefix_embed", "prefix_lm", "suffix_step"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate the public PI0.5 minimal fixture.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    return parser.parse_args()


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object.")
    return value


def require_keys(payload: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise KeyError(f"{context} is missing keys: {missing}")


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def validate_shape(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{context}.shape must be a list.")
    shape = [int(item) for item in value]
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"{context}.shape must contain positive dimensions.")
    return shape


def validate_numeric_list(value: Any, context: str) -> list[int | float]:
    if not isinstance(value, list):
        raise TypeError(f"{context}.data must be a list.")
    data: list[int | float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{context}.data contains a non-numeric value: {item!r}")
        if not math.isfinite(float(item)):
            raise ValueError(f"{context}.data contains a non-finite value: {item!r}")
        data.append(item)
    return data


def validate_tensor(value: Any, context: str, *, dtype: str, rank: int, last_dim: int | None = None) -> dict[str, Any]:
    tensor = require_mapping(value, context)
    require_keys(tensor, ["shape", "dtype", "data"], context)

    shape = validate_shape(tensor["shape"], context)
    if len(shape) != rank:
        raise ValueError(f"{context}.shape must have rank {rank}, got {shape}.")
    if last_dim is not None and shape[-1] != last_dim:
        raise ValueError(f"{context}.shape[-1] must be {last_dim}, got {shape[-1]}.")
    if tensor["dtype"] != dtype:
        raise ValueError(f"{context}.dtype must be {dtype}, got {tensor['dtype']!r}.")

    data = validate_numeric_list(tensor["data"], context)
    if len(data) != product(shape):
        raise ValueError(f"{context}.data has {len(data)} values, but shape {shape} needs {product(shape)}.")
    if dtype == "uint8" and any(int(item) != item or item < 0 or item > 255 for item in data):
        raise ValueError(f"{context}.data must contain uint8 values.")

    return {"shape": shape, "dtype": dtype, "numel": len(data)}


def validate_runtime_contract(value: Any) -> dict[str, Any]:
    contract = require_mapping(value, "runtime_contract")
    require_keys(contract, sorted(REQUIRED_RUNTIME_CONTRACT), "runtime_contract")
    for key, required in REQUIRED_RUNTIME_CONTRACT.items():
        if contract[key] != required:
            raise ValueError(f"runtime_contract.{key} must be {required!r}, got {contract[key]!r}.")
    return contract


def validate_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    require_keys(
        payload,
        ["schema_version", "license", "source", "preset", "repo_id", "model_repo_id", "runtime_contract", "sample"],
        "fixture",
    )
    if payload["schema_version"] != 1:
        raise ValueError(f"schema_version must be 1, got {payload['schema_version']!r}.")
    if payload["source"] != "synthetic_public_fixture":
        raise ValueError("source must be synthetic_public_fixture.")

    contract = validate_runtime_contract(payload["runtime_contract"])
    sample = require_mapping(payload["sample"], "sample")
    require_keys(sample, ["sample_id", "episode_index", "frame_index", "timestamp_s", "observation", "target"], "sample")

    observation = require_mapping(sample["observation"], "sample.observation")
    require_keys(observation, ["language", "state", "images"], "sample.observation")

    language = require_mapping(observation["language"], "sample.observation.language")
    require_keys(language, ["task", "tokenizer_max_length"], "sample.observation.language")
    if not isinstance(language["task"], str) or not language["task"].strip():
        raise ValueError("sample.observation.language.task must be a non-empty string.")
    if language["tokenizer_max_length"] != contract["tokenizer_max_length"]:
        raise ValueError("sample.observation.language.tokenizer_max_length does not match runtime_contract.")

    state = validate_tensor(observation["state"], "sample.observation.state", dtype="float32", rank=1, last_dim=14)
    images = require_mapping(observation["images"], "sample.observation.images")
    if not images:
        raise ValueError("sample.observation.images must contain at least one image.")
    image_summaries = {
        key: validate_tensor(value, f"sample.observation.images.{key}", dtype="uint8", rank=3)
        for key, value in sorted(images.items())
    }

    target = require_mapping(sample["target"], "sample.target")
    require_keys(target, ["action_chunk"], "sample.target")
    action = validate_tensor(
        sample["target"]["action_chunk"],
        "sample.target.action_chunk",
        dtype="float32",
        rank=2,
        last_dim=14,
    )
    if action["shape"][0] > contract["action_chunk_shape"][1]:
        raise ValueError("sample.target.action_chunk has more timesteps than the runtime action chunk.")

    return {
        "fixture": str(DEFAULT_FIXTURE),
        "preset": payload["preset"],
        "repo_id": payload["repo_id"],
        "model_repo_id": payload["model_repo_id"],
        "sample_id": sample["sample_id"],
        "task": language["task"],
        "state": state,
        "images": image_summaries,
        "action_chunk": action,
        "runtime_contract": contract,
    }


def main() -> None:
    args = parse_args()
    payload = require_mapping(json.loads(args.fixture.read_text()), "fixture")
    summary = validate_fixture(payload)
    summary["fixture"] = str(args.fixture)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
