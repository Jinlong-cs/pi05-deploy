#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local PI0.5 Orin runtime stack.")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    _ = parse_args()

    result = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "lerobot_installed": has_module("lerobot"),
        "transformers_installed": has_module("transformers"),
        "pi05_config_module": has_module("lerobot.policies.pi05.configuration_pi05"),
        "pi05_model_module": has_module("lerobot.policies.pi05.modeling_pi05"),
    }

    if result["lerobot_installed"]:
        result["lerobot_version"] = importlib.metadata.version("lerobot")
    if result["transformers_installed"]:
        result["transformers_version"] = importlib.metadata.version("transformers")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
