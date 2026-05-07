#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally benchmark a TensorRT engine via trtexec.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument("--profiling-verbosity", default="detailed")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--duration", type=int, default=0)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        "trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        "--builderOptimizationLevel=5",
        f"--profilingVerbosity={args.profiling_verbosity}",
    ]
    if args.precision == "fp16":
        build_cmd.append("--fp16")
    else:
        build_cmd.extend(["--int8", "--fp16"])
    if args.timing_cache is not None:
        build_cmd.append(f"--timingCacheFile={args.timing_cache}")
    if args.skip_inference:
        build_cmd.append("--skipInference")
    build_cmd.extend(args.extra_arg)
    run(build_cmd)

    if args.benchmark:
        benchmark_cmd = [
            "trtexec",
            f"--loadEngine={args.engine}",
            f"--warmUp={args.warmup_ms}",
            f"--iterations={args.iterations}",
            f"--duration={args.duration}",
        ]
        if args.report_dir is not None:
            benchmark_cmd.extend(
                [
                    f"--exportTimes={args.report_dir / (args.engine.stem + '_times.json')}",
                    f"--exportProfile={args.report_dir / (args.engine.stem + '_profile.json')}",
                    f"--dumpLayerInfo",
                ]
            )
        run(benchmark_cmd)


if __name__ == "__main__":
    main()
