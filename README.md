# pi0.5-orin

Deploy-only PI0.5 runtime and TensorRT tooling for Jetson Orin.

This repository keeps the deployment path only:

- open-loop eval and latency benchmark
- ONNX export for PI0.5 prefix/suffix stages
- TensorRT engine build helpers
- INT8 PTQ and QDQ utilities for Orin deployment

It intentionally leaves out reports, profiling summaries, temporary experiments, datasets, weights, and generated engines.

## Scope

`pi0.5-orin` is a small deployment repository for running `lerobot/pi05_base` on Orin-class devices. It is not a training repo and it is not a full copy of the original `tinyaction/pi0.5` workspace.

Current runtime paths:

- `pytorch`: eager baseline for bring-up and correctness checks
- `trt_fp16`: staged TensorRT runtime
- `trt_int8`: staged TensorRT runtime with PTQ/QDQ engines

## Requirements

- Python `>=3.10`
- NVIDIA GPU for real deployment runs
- Jetson Orin + JetPack 6 for the intended target path

Recommended Jetson runtime mode:

- `MAXN`
- `jetson_clocks`

## Repository Layout

- `src/pi05_orin/`: runtime, presets, TensorRT runner, wrappers
- `scripts/`: setup, download, eval, benchmark, export, TensorRT build
- `docs/deploy.md`: concise runtime notes

## Install

Standard Python environment:

```bash
bash scripts/bootstrap_env.sh
```

Jetson Orin / JetPack 6:

```bash
bash scripts/install_jetson_pi05_stack.sh
```

Validate the install:

```bash
.venv/bin/python scripts/setup_pi05_stack.py --validate
```

## Runtime Layout

The optimized Orin path is split into three TensorRT stages plus a CUDA-graph suffix loop:

1. `prefix_embed`
2. `prefix_lm`
3. `suffix_step`
4. repeated suffix denoise steps via CUDA Graph replay

## Quick Start

```bash
.venv/bin/python scripts/download_dataset.py --preset pi05_aloha_public --mode metadata
.venv/bin/python scripts/make_fixed_eval_split.py --preset pi05_aloha_public
.venv/bin/python scripts/download_pi05_assets.py --preset pi05_aloha_public

.venv/bin/python scripts/eval_open_loop_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend pytorch

.venv/bin/python scripts/benchmark_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend trt_int8
```

## TensorRT

Main deploy-side scripts:

- `scripts/capture_pi05_trt_inputs.py`
- `scripts/export_pi05_prefix_onnx.py`
- `scripts/export_pi05_suffix_onnx.py`
- `scripts/build_trt_engine.py`
- `scripts/build_trt_ptq_engine.py`
- `scripts/quantize_suffix_onnx_qdq.py`

See [`docs/deploy.md`](docs/deploy.md) for the staged runtime layout and Jetson-specific notes.

## Defaults

- dataset preset: `lerobot/aloha_sim_insertion_human_image`
- model preset: `lerobot/pi05_base`
- batch size: `1`
- tokenizer max length: `200`
- inference steps: `10`

## Non-Goals

This repository does not include:

- training code
- experiment reports
- profiling outputs
- generated engines or ONNX exports
- model weights or datasets
