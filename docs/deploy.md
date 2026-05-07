# Deploy Notes

This repository keeps only the deployment-facing PI0.5 stack for Jetson Orin.

Supported runtime paths:

- `pytorch`: eager baseline for bring-up and correctness checks
- `trt_fp16`: staged TensorRT runtime
- `trt_int8`: staged TensorRT runtime with PTQ/QDQ engines

Default optimized runtime layout:

- `prefix_embed`: TensorRT engine
- `prefix_lm`: TensorRT engine
- `suffix_step`: TensorRT engine
- `suffix loop`: CUDA Graph replay over 10 denoise steps

Default public preset:

- dataset: `lerobot/aloha_sim_insertion_human_image`
- model: `lerobot/pi05_base`
- batch size: `1`
- tokenizer max length: `200`
- action chunk: `50`
- inference steps: `10`

Jetson notes:

- run in `MAXN`
- pin clocks with `jetson_clocks`
- TensorRT Python bindings are expected from the JetPack system install, not from PyPI

This repo does not include generated engines, ONNX exports, model weights, datasets, reports, or profiling outputs.
