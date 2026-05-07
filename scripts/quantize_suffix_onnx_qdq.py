#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static QDQ-quantized suffix_step ONNX model.")
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--output-onnx", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-steps-per-sample", type=int)
    parser.add_argument("--calibrate-method", choices=["minmax", "entropy"], default="minmax")
    parser.add_argument("--op-types", nargs="+", default=["MatMul", "Gemm"])
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--force-quantize-no-input-check", action="store_true")
    parser.add_argument("--activation-symmetric", action="store_true")
    parser.add_argument("--weight-symmetric", action="store_true")
    parser.add_argument("--add-qdq-pair-to-weight", action="store_true")
    parser.add_argument("--dedicated-qdq-pair", action="store_true")
    return parser.parse_args()


def _load_tensor_payload(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {path}, got {type(payload).__name__}.")
    return payload


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().contiguous().cpu().numpy()


class SuffixStepQdqDataReader(CalibrationDataReader):
    def __init__(
        self,
        *,
        capture_root: Path,
        limit_samples: int | None,
        limit_steps_per_sample: int | None,
    ) -> None:
        self.sample_dirs = sorted(path for path in capture_root.iterdir() if path.is_dir())
        if limit_samples is not None:
            self.sample_dirs = self.sample_dirs[:limit_samples]
        self.limit_steps_per_sample = limit_steps_per_sample
        self.sample_index = 0
        self.step_index = 0
        self.current_prefix: dict[str, np.ndarray] | None = None
        self.current_steps: list[dict[str, Any]] = []
        self.current_dt: np.ndarray | None = None
        self.yielded_batches = 0
        self.num_sample_dirs = len(self.sample_dirs)
        self.num_suffix_steps = 0
        for sample_dir in self.sample_dirs:
            suffix_loop_path = sample_dir / "suffix_loop.pt"
            if suffix_loop_path.exists():
                payload = torch.load(suffix_loop_path, map_location="cpu")
                step_count = len(payload["steps"])
                if self.limit_steps_per_sample is not None:
                    step_count = min(step_count, self.limit_steps_per_sample)
                self.num_suffix_steps += step_count
            else:
                self.num_suffix_steps += 1

    def _load_current_sample(self) -> bool:
        if self.sample_index >= len(self.sample_dirs):
            return False

        sample_dir = self.sample_dirs[self.sample_index]
        prefix_payload = _load_tensor_payload(sample_dir / "prefix.pt")
        self.current_prefix = {
            name: _to_numpy(tensor)
            for name, tensor in prefix_payload.items()
        }

        suffix_loop_path = sample_dir / "suffix_loop.pt"
        if suffix_loop_path.exists():
            payload = torch.load(suffix_loop_path, map_location="cpu")
            steps = list(payload["steps"])
            if self.limit_steps_per_sample is not None:
                steps = steps[: self.limit_steps_per_sample]
            self.current_steps = steps
            dt_value = payload.get("dt")
            if dt_value is None:
                dt_value = -1.0 / max(1, len(self.current_steps))
        else:
            suffix_payload = _load_tensor_payload(sample_dir / "suffix_step.pt")
            self.current_steps = [
                {
                    "x_t": suffix_payload["x_t"],
                    "timestep": suffix_payload["timestep"],
                }
            ]
            dt_value = suffix_payload.get("dt")
            if isinstance(dt_value, torch.Tensor):
                dt_value = float(dt_value.reshape(-1)[0].item())
            elif dt_value is None:
                dt_value = -1.0

        self.current_dt = np.asarray([float(dt_value)], dtype=np.float32)
        self.step_index = 0
        return True

    def get_next(self) -> dict[str, np.ndarray] | None:
        while True:
            if self.current_prefix is None or self.step_index >= len(self.current_steps):
                if not self._load_current_sample():
                    return None

            if self.step_index >= len(self.current_steps):
                self.sample_index += 1
                self.current_prefix = None
                continue

            assert self.current_prefix is not None
            assert self.current_dt is not None
            step_payload = self.current_steps[self.step_index]
            self.step_index += 1

            if self.step_index >= len(self.current_steps):
                self.sample_index += 1
                next_prefix = None
            else:
                next_prefix = self.current_prefix

            batch = {
                "prefix_pad_masks": self.current_prefix["prefix_pad_masks"],
                "x_t": _to_numpy(step_payload["x_t"]),  # type: ignore[arg-type]
                "timestep": _to_numpy(step_payload["timestep"]),  # type: ignore[arg-type]
                "dt": self.current_dt,
            }
            for name, value in self.current_prefix.items():
                if name.startswith("cache_"):
                    batch[name] = value

            self.current_prefix = next_prefix
            self.yielded_batches += 1
            return batch

    def rewind(self) -> None:
        self.sample_index = 0
        self.step_index = 0
        self.current_prefix = None
        self.current_steps = []
        self.current_dt = None
        self.yielded_batches = 0

    def describe(self) -> dict[str, object]:
        return {
            "capture_root": str(self.capture_root),
            "num_sample_dirs": self.num_sample_dirs,
            "num_suffix_steps": self.num_suffix_steps,
            "yielded_batches": self.yielded_batches,
        }

    @property
    def capture_root(self) -> Path:
        return self.sample_dirs[0].parent if self.sample_dirs else Path()


def calibration_method_from_name(name: str) -> CalibrationMethod:
    if name == "minmax":
        return CalibrationMethod.MinMax
    if name == "entropy":
        return CalibrationMethod.Entropy
    raise ValueError(f"Unsupported calibration method: {name}")


def summarize_qdq_model(path: Path) -> dict[str, object]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    node_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "path": str(path),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "quantize_linear_count": node_counts.get("QuantizeLinear", 0),
        "dequantize_linear_count": node_counts.get("DequantizeLinear", 0),
        "op_type_counts": dict(sorted(node_counts.items())),
    }


def main() -> None:
    args = parse_args()
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or args.output_onnx.with_suffix(".qdq_summary.json")

    reader = SuffixStepQdqDataReader(
        capture_root=args.capture_root,
        limit_samples=args.limit_samples,
        limit_steps_per_sample=args.limit_steps_per_sample,
    )
    extra_options = {
        "ActivationSymmetric": args.activation_symmetric,
        "WeightSymmetric": args.weight_symmetric,
        "ForceQuantizeNoInputCheck": args.force_quantize_no_input_check,
        "AddQDQPairToWeight": args.add_qdq_pair_to_weight,
        "DedicatedQDQPair": args.dedicated_qdq_pair,
        "CalibTensorRangeSymmetric": args.activation_symmetric,
    }

    quantize_static(
        model_input=args.source_onnx,
        model_output=args.output_onnx,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=args.op_types,
        per_channel=args.per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=calibration_method_from_name(args.calibrate_method),
        extra_options=extra_options,
    )

    summary = {
        "source_onnx": str(args.source_onnx),
        "output_onnx": str(args.output_onnx),
        "capture_root": str(args.capture_root),
        "calibrate_method": args.calibrate_method,
        "op_types": args.op_types,
        "per_channel": args.per_channel,
        "extra_options": extra_options,
        "reader": reader.describe(),
        "qdq_model": summarize_qdq_model(args.output_onnx),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
