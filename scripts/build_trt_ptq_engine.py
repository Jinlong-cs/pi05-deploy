#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt

from pi05_orin.trt_ptq import (
    make_calibration_stream,
    make_calibrator,
    write_ptq_summary,
)


TRT_LOGGER_SEVERITY = {
    "internal_error": trt.Logger.INTERNAL_ERROR,
    "error": trt.Logger.ERROR,
    "warning": trt.Logger.WARNING,
    "info": trt.Logger.INFO,
    "verbose": trt.Logger.VERBOSE,
}

TRT_PROFILING_VERBOSITY = {
    "layer_names_only": trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
    "detailed": trt.ProfilingVerbosity.DETAILED,
    "none": trt.ProfilingVerbosity.NONE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real PTQ TensorRT engine via TensorRT Python calibrator.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--stage", choices=["prefix_embed", "prefix_lm", "suffix_step", "suffix_unrolled"], required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["minmax", "entropy2"], default="minmax")
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    parser.add_argument("--profiling-verbosity", choices=sorted(TRT_PROFILING_VERBOSITY), default="detailed")
    parser.add_argument("--workspace-gib", type=int, default=8)
    parser.add_argument("--logger-severity", choices=sorted(TRT_LOGGER_SEVERITY), default="info")
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def load_onnx_network(*, onnx_path: Path, logger: trt.Logger) -> tuple[trt.Builder, trt.INetworkDefinition, trt.OnnxParser]:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = []
        for index in range(parser.num_errors):
            errors.append(str(parser.get_error(index)))
        raise RuntimeError(f"Failed to parse ONNX {onnx_path}:\n" + "\n".join(errors))
    return builder, network, parser


def main() -> None:
    args = parse_args()
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.calibration_cache or args.engine.with_suffix(".calibration.cache")
    summary_path = args.summary_json or args.engine.with_suffix(".ptq_summary.json")

    logger = trt.Logger(TRT_LOGGER_SEVERITY[args.logger_severity])
    builder, network, _ = load_onnx_network(onnx_path=args.onnx, logger=logger)
    input_names = [network.get_input(index).name for index in range(network.num_inputs)]

    stream = make_calibration_stream(args.stage, args.capture_root)
    calibrator = make_calibrator(
        trt=trt,
        algorithm=args.algorithm,
        input_names=input_names,
        stream=stream,
        cache_file=cache_path,
        log_every=args.log_every,
    )

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator
    config.profiling_verbosity = TRT_PROFILING_VERBOSITY[args.profiling_verbosity]
    config.builder_optimization_level = args.builder_optimization_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib * (1 << 30))

    print(f"Building PTQ TensorRT engine for stage={args.stage}")
    print(f"ONNX: {args.onnx}")
    print(f"Engine: {args.engine}")
    print(f"Calibration root: {args.capture_root}")
    print(f"Input names: {input_names}")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT builder returned no serialized engine.")
    args.engine.write_bytes(bytes(serialized_engine))
    write_ptq_summary(summary_path, stage=args.stage, algorithm=args.algorithm, stream=stream, input_names=input_names)
    print(f"Wrote engine to {args.engine}")
    print(f"Wrote PTQ summary to {summary_path}")


if __name__ == "__main__":
    main()
