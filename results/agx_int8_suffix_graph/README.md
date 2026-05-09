# AGX INT8 Suffix-Graph Baseline

Final AGX JSON results for the current PI0.5 Orin deployment baseline.

Environment:

- Device: NVIDIA Jetson AGX Orin
- Mode: `MAXN + jetson_clocks`
- Dataset: `lerobot/aloha_sim_insertion_human_image`
- Model: `lerobot/pi05_base`
- Batch size: `1`
- Tokenizer max length: `200`
- Action chunk: `50 x 14`
- Internal action latent: `50 x 32`
- Inference steps: `10`
- Benchmark warmup / measure batches: `5 / 20`

Latency:

| Runtime | Mean policy | Mean E2E | P95 E2E |
| --- | ---: | ---: | ---: |
| TRT real PTQ INT8 | 198.88 ms | 202.15 ms | 202.32 ms |
| TRT real PTQ INT8 + suffix CUDA Graph | 185.92 ms | 189.56 ms | 189.67 ms |
| TRT real PTQ INT8 + prefix_lm+suffix CUDA Graph | 183.20 ms | 186.52 ms | 186.74 ms |

Default optimized stage means:

| Stage | Mean latency |
| --- | ---: |
| `prefix_embed.engine` | 15.70 ms |
| `prefix_lm.engine` | 80.00 ms |
| `suffix_step.engine` naive 10-step loop | 107.14 ms |
| `suffix_step.engine` CUDA Graph 10-step replay | 88.81 ms |

Open-loop full-eval metrics for `trt_int8 + suffix graph`:

| Metric | Value |
| --- | ---: |
| normalized action MSE | 0.3166 |
| normalized action L1 | 0.4768 |
| normalized first-action MSE | 0.3102 |
| normalized first-action L1 | 0.4722 |

Files:

- `trt_int8_suffix_graph_latency.json`
- `trt_int8_prefix_lm_suffix_graph_latency.json`
- `trt_int8_stage_split.json`
- `trt_int8_suffix_graph_profile.json`
- `trt_int8_suffix_vs_full_graph_seed_parity.json`
- `trt_int8_open_loop_full_eval.json`

Caveats:

- The full graph path is kept as an opt-in experiment, not the default runtime.
- Open-loop action error is not a closed-loop robot success-rate metric.
- The current suffix INT8 engine still has FP16/FP32 fallback paths, so this is
  a practical PTQ deployment baseline rather than a pure INT8 implementation.
