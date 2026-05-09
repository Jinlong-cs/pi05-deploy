#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local PI0.5 Orin runtime stack.")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when required runtime components are missing.")
    return parser.parse_args()


def has_module(name: str) -> bool:
    checked = []
    for part in name.split("."):
        checked.append(part)
        if importlib.util.find_spec(".".join(checked)) is None:
            return False
    return True


def main() -> None:
    args = parse_args()

    result = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_installed": has_module("torch"),
        "torch_cuda_available": False,
        "tensorrt_installed": has_module("tensorrt"),
        "trtexec_available": shutil.which("trtexec") is not None,
        "lerobot_installed": has_module("lerobot"),
        "transformers_installed": has_module("transformers"),
        "pi05_config_module": has_module("lerobot.policies.pi05.configuration_pi05"),
        "pi05_model_module": has_module("lerobot.policies.pi05.modeling_pi05"),
    }

    if result["torch_installed"]:
        import torch

        result["torch_version"] = torch.__version__
        result["torch_cuda_available"] = bool(torch.cuda.is_available())
    if result["lerobot_installed"]:
        result["lerobot_version"] = importlib.metadata.version("lerobot")
    if result["transformers_installed"]:
        result["transformers_version"] = importlib.metadata.version("transformers")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict:
        required = [
            "torch_installed",
            "lerobot_installed",
            "transformers_installed",
            "pi05_config_module",
            "pi05_model_module",
        ]
        missing = [name for name in required if not result[name]]
        if missing:
            raise SystemExit(f"Missing required components: {', '.join(missing)}")


if __name__ == "__main__":
    main()
