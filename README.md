# PI0.5 Orin Deploy

Deployment-oriented PI0.5 runtime and TensorRT tooling for NVIDIA Jetson AGX
Orin.

This repository keeps the small deploy layer needed to run `lerobot/pi05_base`
with:

- PyTorch eager inference for bring-up and correctness checks.
- Partitioned TensorRT FP16 / INT8 inference for `prefix_embed`, `prefix_lm`,
  and `suffix_step`.
- Real TensorRT PTQ calibration utilities for the three exported stages.
- QDQ helper utilities for suffix quantization experiments.
- Open-loop evaluation, latency benchmark, and final AGX baseline JSON results.

Upstream model package: [huggingface.co/lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base).

## File Structure

```text
pi0.5-orin/
├── docs/
│   ├── deploy.md                         # Command reference for the deploy flow
│   └── pi05_deployment_pipeline_agx_orin.png
├── results/
│   └── agx_int8_suffix_graph/            # Final AGX INT8 + suffix-graph JSON results
├── scripts/
│   ├── install_jetson_pi05_stack.sh
│   ├── setup_pi05_stack.py
│   ├── download_dataset.py
│   ├── make_fixed_eval_split.py
│   ├── download_pi05_assets.py
│   ├── eval_open_loop_pi05.py
│   ├── benchmark_pi05.py
│   ├── capture_pi05_trt_inputs.py
│   ├── export_pi05_prefix_onnx.py
│   ├── export_pi05_suffix_onnx.py
│   ├── build_trt_engine.py
│   ├── build_trt_ptq_engine.py
│   ├── rewrite_suffix_onnx_fp64_trig_to_fp32.py
│   └── quantize_suffix_onnx_qdq.py
└── src/pi05_orin/                         # Runtime, paths, presets, wrappers
```

Large assets are intentionally not tracked:

- LeRobot datasets and split cache.
- PI0.5 model weights.
- Captured calibration tensors.
- ONNX exports and TensorRT engines.
- Intermediate profiling outputs.

## Runtime Layout

The optimized TensorRT path uses three engines and one CUDA Graph wrapped loop:

```text
LeRobot sample + preprocessor
  -> prefix_embed.engine
  -> prefix_lm.engine
  -> suffix_step.engine replayed for 10 denoise steps with CUDA Graph
  -> host-side action crop / postprocess
```

Default public preset:

- dataset: `lerobot/aloha_sim_insertion_human_image`
- model: `lerobot/pi05_base`
- batch size: `1`
- tokenizer max length: `200`
- prefix length: `456`
- action latent: `[1, 50, 32]`
- final action chunk: `[1, 50, 14]`
- inference steps: `10`

Pipeline figure:

- [`docs/pi05_deployment_pipeline_agx_orin.png`](docs/pi05_deployment_pipeline_agx_orin.png)

## Environment Setup

Use Python 3.10+.

For Jetson Orin / JetPack 6, use the Jetson bootstrap script. It keeps the
NVIDIA-provided TensorRT Python bindings from the system install and installs a
Jetson-compatible PyTorch stack.

```bash
bash scripts/install_jetson_pi05_stack.sh
```

For a standard Python environment used only for syntax checks or CPU-side
tooling:

```bash
bash scripts/bootstrap_env.sh
```

Validate the local stack:

```bash
PYTHONPATH=src .venv/bin/python scripts/setup_pi05_stack.py --validate
```

On AGX benchmark runs, use:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Data and Model Preparation

```bash
PYTHONPATH=src .venv/bin/python scripts/download_dataset.py \
  --preset pi05_aloha_public \
  --mode metadata

PYTHONPATH=src .venv/bin/python scripts/make_fixed_eval_split.py \
  --preset pi05_aloha_public

PYTHONPATH=src .venv/bin/python scripts/download_pi05_assets.py \
  --preset pi05_aloha_public
```

Expected local layout after preparation:

```text
data/
├── raw/lerobot__aloha_sim_insertion_human_image/
└── splits/lerobot__aloha_sim_insertion_human_image/

artifacts/models/lerobot__pi05_base/
```

## PyTorch Eager Bring-Up

Run a short open-loop check before exporting TensorRT engines:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_open_loop_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend pytorch \
  --limit-batches 1
```

Latency benchmark:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend pytorch \
  --warmup-batches 5 \
  --measure-batches 20
```

## TensorRT Export and Engine Build

Capture representative tensors for ONNX export and INT8 calibration:

```bash
PYTHONPATH=src .venv/bin/python scripts/capture_pi05_trt_inputs.py \
  --preset pi05_aloha_public \
  --split train \
  --num-samples 40 \
  --capture-suffix-loop \
  --device cuda
```

Export the three ONNX partitions:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_pi05_prefix_onnx.py \
  --preset pi05_aloha_public \
  --stage all \
  --device cuda

PYTHONPATH=src .venv/bin/python scripts/export_pi05_suffix_onnx.py \
  --preset pi05_aloha_public \
  --device cuda
```

Build FP16 engines with `trtexec`:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_trt_engine.py \
  --onnx outputs/trt_onnx/pi05_aloha_public/prefix_embed.onnx \
  --engine outputs/trt_engines/pi05_aloha_public/fp16/prefix_embed.engine \
  --precision fp16

PYTHONPATH=src .venv/bin/python scripts/build_trt_engine.py \
  --onnx outputs/trt_onnx/pi05_aloha_public/prefix_lm.onnx \
  --engine outputs/trt_engines/pi05_aloha_public/fp16/prefix_lm.engine \
  --precision fp16

PYTHONPATH=src .venv/bin/python scripts/build_trt_engine.py \
  --onnx outputs/trt_onnx/pi05_aloha_public/suffix_step.onnx \
  --engine outputs/trt_engines/pi05_aloha_public/fp16/suffix_step.engine \
  --precision fp16
```

Build real PTQ INT8 engines with TensorRT Python calibrators:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_trt_ptq_engine.py \
  --stage prefix_embed \
  --onnx outputs/trt_onnx/pi05_aloha_public/prefix_embed.onnx \
  --engine outputs/trt_engines/pi05_aloha_public/int8/prefix_embed.engine \
  --capture-root outputs/trt_captures/pi05_aloha_public

PYTHONPATH=src .venv/bin/python scripts/build_trt_ptq_engine.py \
  --stage prefix_lm \
  --onnx outputs/trt_onnx/pi05_aloha_public/prefix_lm.onnx \
  --engine outputs/trt_engines/pi05_aloha_public/int8/prefix_lm.engine \
  --capture-root outputs/trt_captures/pi05_aloha_public

PYTHONPATH=src .venv/bin/python scripts/build_trt_ptq_engine.py \
  --stage suffix_step \
  --onnx outputs/trt_onnx/pi05_aloha_public/suffix_step.onnx \
  --engine outputs/trt_engines/pi05_aloha_public/int8/suffix_step.engine \
  --capture-root outputs/trt_captures/pi05_aloha_public \
  --log-every 100
```

The simpler `build_trt_engine.py --precision int8` path enables TensorRT
`--int8 --fp16`, but it is not the same as real PTQ because no calibrator is
attached. Use `build_trt_ptq_engine.py` for the main INT8 baseline.

## TensorRT Runtime Benchmark

Run INT8 runtime with the default suffix CUDA Graph path:

```bash
PYTHONPATH=src PI05_TRT_SUFFIX_GRAPH=1 .venv/bin/python scripts/benchmark_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend trt_int8 \
  --warmup-batches 5 \
  --measure-batches 20 \
  --output outputs/benchmarks/pi05_trt_int8_suffix_graph_5w20m.json
```

Use `--engine-root /path/to/pi05_aloha_public` when the engines are stored
outside the default `outputs/trt_engines/<preset>/` directory.

Run open-loop evaluation:

```bash
PYTHONPATH=src PI05_TRT_SUFFIX_GRAPH=1 .venv/bin/python scripts/eval_open_loop_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend trt_int8 \
  --output outputs/eval/pi05_trt_int8_suffix_graph_open_loop.json
```

Disable suffix graph for an A/B runtime check:

```bash
PYTHONPATH=src PI05_TRT_SUFFIX_GRAPH=0 .venv/bin/python scripts/benchmark_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend trt_int8
```

## Current AGX Baseline

Final AGX JSON outputs are kept under
[`results/agx_int8_suffix_graph/`](results/agx_int8_suffix_graph/).

Measured on Jetson AGX Orin with `MAXN + jetson_clocks`, batch size 1,
`num_inference_steps=10`, `warmup=5`, `measure=20`.

| Runtime | Mean policy | Mean E2E | P95 E2E | Notes |
| --- | ---: | ---: | ---: | --- |
| PyTorch eager baseline | 1744.72 ms | 1748.02 ms | n/a | Original full policy path |
| TRT FP16 | 208.65 ms | 211.91 ms | n/a | 3 staged engines |
| TRT real PTQ INT8 | 198.88 ms | 202.15 ms | 202.32 ms | 3 PTQ engines |
| TRT real PTQ INT8 + suffix CUDA Graph | 185.92 ms | 189.56 ms | 189.67 ms | Default optimized path |
| TRT real PTQ INT8 + prefix_lm+suffix CUDA Graph | 183.20 ms | 186.52 ms | 186.74 ms | Opt-in experiment, not default |

Default optimized stage means:

| Stage | Mean latency |
| --- | ---: |
| Host preprocess | 3.64 ms |
| `prefix_embed.engine` | 15.70 ms |
| `prefix_lm.engine` | 80.00 ms |
| `suffix_step.engine` naive 10-step loop | 107.14 ms |
| `suffix_step.engine` CUDA Graph 10-step replay | 88.81 ms |

Open-loop full-eval metrics for the INT8 graph path:

| Metric | Value |
| --- | ---: |
| normalized action MSE | 0.3166 |
| normalized action L1 | 0.4768 |
| normalized first-action MSE | 0.3102 |
| normalized first-action L1 | 0.4722 |

Accuracy caveat:

- These are open-loop action errors, not closed-loop robot success rates.
- Current suffix INT8 still contains substantial FP16/FP32 fallback paths in
  TensorRT inspector output, so the INT8 work should be read as a stable PTQ
  baseline, not a pure INT8 implementation.

## Optional QDQ Suffix Utilities

The QDQ path is kept as an experimental suffix quantization helper:

```bash
PYTHONPATH=src .venv/bin/python scripts/rewrite_suffix_onnx_fp64_trig_to_fp32.py \
  --source-onnx outputs/trt_onnx/pi05_aloha_public/suffix_step.onnx \
  --output-onnx outputs/trt_onnx/pi05_aloha_public/suffix_step_fp32_trig.onnx

PYTHONPATH=src .venv/bin/python scripts/quantize_suffix_onnx_qdq.py \
  --source-onnx outputs/trt_onnx/pi05_aloha_public/suffix_step_fp32_trig.onnx \
  --output-onnx outputs/trt_onnx/pi05_aloha_public/suffix_step_qdq.onnx \
  --capture-root outputs/trt_captures/pi05_aloha_public \
  --op-types MatMul
```

Do not treat QDQ suffix output as the default runtime until latency and
open-loop drift are revalidated for the exact engine being used.

## License

This repository is released as a deployment wrapper. Upstream LeRobot, PI0.5,
datasets, and NVIDIA TensorRT components retain their original licenses.
