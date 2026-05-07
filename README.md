# pi0.5-orin

`pi0.5-orin` is a small deploy-only repository for running PI0.5 (`lerobot/pi05_base`) on Jetson Orin.

It keeps only the Orin inference path:

- open-loop eval and latency benchmark
- ONNX export for PI0.5 prefix/suffix stages
- TensorRT FP16 / INT8 engine build helpers
- INT8 PTQ and QDQ utilities

Supported runtimes:

- `pytorch`: eager baseline for bring-up and correctness checks
- `trt_fp16`: staged TensorRT runtime
- `trt_int8`: staged TensorRT runtime

Optimized runtime layout:

1. `prefix_embed`
2. `prefix_lm`
3. `suffix_step`
4. CUDA Graph replay over repeated suffix denoise steps

## Requirements

- Python `>=3.10`
- NVIDIA GPU for real deployment runs
- Jetson Orin + JetPack 6 for the intended target path
- recommended Jetson mode: `MAXN + jetson_clocks`

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

## Quick Start

Default public preset:

- dataset: `lerobot/aloha_sim_insertion_human_image`
- model: `lerobot/pi05_base`
- batch size: `1`

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

See [`docs/deploy.md`](docs/deploy.md) for stage layout and Jetson-specific notes.

## Repository Layout

- `src/pi05_orin/`: runtime, presets, TensorRT runner, wrappers
- `scripts/`: setup, download, eval, benchmark, export, TensorRT build
- `docs/deploy.md`: concise runtime notes

Main TensorRT scripts:

- `scripts/capture_pi05_trt_inputs.py`
- `scripts/export_pi05_prefix_onnx.py`
- `scripts/export_pi05_suffix_onnx.py`
- `scripts/build_trt_engine.py`
- `scripts/build_trt_ptq_engine.py`
- `scripts/quantize_suffix_onnx_qdq.py`

## Not Included

This repository intentionally does not include:

- training code
- experiment reports
- profiling outputs
- generated engines or ONNX exports
- model weights or datasets
