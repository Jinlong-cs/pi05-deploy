# Deploy Command Reference

This page is the short command reference for the PI0.5 Orin deployment flow.
The README contains the longer explanation and the public latency summary.

## Runtime Contract

Default preset:

- dataset: `lerobot/aloha_sim_insertion_human_image`
- model: `lerobot/pi05_base`
- backend choices: `pytorch`, `trt_fp16`, `trt_int8`
- TensorRT stages: `prefix_embed`, `prefix_lm`, `suffix_step`
- suffix denoise steps: `10`
- default graph setting: `PI05_TRT_SUFFIX_GRAPH=1`

Engine lookup order for a stage such as `prefix_lm`:

```text
outputs/trt_engines/pi05_aloha_public/int8/prefix_lm.engine
outputs/trt_engines/pi05_aloha_public/int8/prefix_lm.plan
outputs/trt_engines/pi05_aloha_public/prefix_lm_int8.engine
outputs/trt_engines/pi05_aloha_public/prefix_lm_int8.plan
outputs/trt_engines/pi05_aloha_public/prefix_lm.engine
outputs/trt_engines/pi05_aloha_public/prefix_lm.plan
```

Override paths with:

```bash
export PI05_DATA_DIR=/path/to/data
export PI05_OUTPUTS_DIR=/path/to/outputs
export PI05_TRT_ENGINES_DIR=/path/to/trt_engines
```

`benchmark_pi05.py` and `eval_open_loop_pi05.py` also accept
`--engine-root /path/to/pi05_aloha_public` for one-off engine selection.

## 1. Install

Jetson Orin:

```bash
bash scripts/install_jetson_pi05_stack.sh
PYTHONPATH=src .venv/bin/python scripts/setup_pi05_stack.py --validate
```

Generic Python environment:

```bash
bash scripts/bootstrap_env.sh
PYTHONPATH=src .venv/bin/python scripts/setup_pi05_stack.py --validate
```

## 2. Prepare Data and Model

```bash
PYTHONPATH=src .venv/bin/python scripts/download_dataset.py \
  --preset pi05_aloha_public \
  --mode metadata

PYTHONPATH=src .venv/bin/python scripts/make_fixed_eval_split.py \
  --preset pi05_aloha_public

PYTHONPATH=src .venv/bin/python scripts/download_pi05_assets.py \
  --preset pi05_aloha_public
```

## 3. PyTorch Bring-Up

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_open_loop_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend pytorch \
  --limit-batches 1
```

## 4. Capture Calibration Inputs

For the default PTQ path, capture prefix inputs and full suffix-loop states:

```bash
PYTHONPATH=src .venv/bin/python scripts/capture_pi05_trt_inputs.py \
  --preset pi05_aloha_public \
  --split train \
  --num-samples 40 \
  --capture-suffix-loop \
  --device cuda
```

For a quick single-sample export smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/capture_pi05_trt_inputs.py \
  --preset pi05_aloha_public \
  --split train \
  --num-samples 1 \
  --capture-suffix-step \
  --device cuda
```

## 5. Export ONNX

```bash
PYTHONPATH=src .venv/bin/python scripts/export_pi05_prefix_onnx.py \
  --preset pi05_aloha_public \
  --stage all \
  --device cuda

PYTHONPATH=src .venv/bin/python scripts/export_pi05_suffix_onnx.py \
  --preset pi05_aloha_public \
  --device cuda
```

Expected outputs:

```text
outputs/trt_onnx/pi05_aloha_public/
├── prefix_embed.onnx
├── prefix_lm.onnx
└── suffix_step.onnx
```

## 6. Build Engines

FP16:

```bash
for stage in prefix_embed prefix_lm suffix_step; do
  PYTHONPATH=src .venv/bin/python scripts/build_trt_engine.py \
    --onnx outputs/trt_onnx/pi05_aloha_public/${stage}.onnx \
    --engine outputs/trt_engines/pi05_aloha_public/fp16/${stage}.engine \
    --precision fp16
done
```

Real PTQ INT8:

```bash
for stage in prefix_embed prefix_lm suffix_step; do
  log_every=20
  if [ "${stage}" = "suffix_step" ]; then log_every=100; fi
  PYTHONPATH=src .venv/bin/python scripts/build_trt_ptq_engine.py \
    --stage ${stage} \
    --onnx outputs/trt_onnx/pi05_aloha_public/${stage}.onnx \
    --engine outputs/trt_engines/pi05_aloha_public/int8/${stage}.engine \
    --capture-root outputs/trt_captures/pi05_aloha_public \
    --log-every ${log_every}
done
```

## 7. Benchmark and Eval

```bash
PYTHONPATH=src PI05_TRT_SUFFIX_GRAPH=1 .venv/bin/python scripts/benchmark_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend trt_int8 \
  --warmup-batches 5 \
  --measure-batches 20

PYTHONPATH=src PI05_TRT_SUFFIX_GRAPH=1 .venv/bin/python scripts/eval_open_loop_pi05.py \
  --preset pi05_aloha_public \
  --device cuda \
  --backend trt_int8
```

## 8. QDQ Suffix Experiment

This is optional and not the default runtime path.

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
