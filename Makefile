.PHONY: install jetson-install validate prepare-data eval-pytorch bench-pytorch capture export-prefix export-suffix bench-trt-int8 eval-trt-int8

PYTHON ?= .venv/bin/python
PYTHONPATH := src
PRESET ?= pi05_aloha_public
DEVICE ?= cuda

install:
	bash scripts/bootstrap_env.sh

jetson-install:
	bash scripts/install_jetson_pi05_stack.sh

validate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/setup_pi05_stack.py --validate

prepare-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/download_dataset.py --preset $(PRESET) --mode metadata
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/make_fixed_eval_split.py --preset $(PRESET)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/download_pi05_assets.py --preset $(PRESET)

eval-pytorch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/eval_open_loop_pi05.py \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--backend pytorch \
		--limit-batches 1

bench-pytorch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_pi05.py \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--backend pytorch \
		--warmup-batches 5 \
		--measure-batches 20

capture:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/capture_pi05_trt_inputs.py \
		--preset $(PRESET) \
		--split train \
		--num-samples 40 \
		--capture-suffix-loop \
		--device $(DEVICE)

export-prefix:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_pi05_prefix_onnx.py \
		--preset $(PRESET) \
		--stage all \
		--device $(DEVICE)

export-suffix:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_pi05_suffix_onnx.py \
		--preset $(PRESET) \
		--device $(DEVICE)

bench-trt-int8:
	PYTHONPATH=$(PYTHONPATH) PI05_TRT_SUFFIX_GRAPH=1 $(PYTHON) scripts/benchmark_pi05.py \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--backend trt_int8 \
		--warmup-batches 5 \
		--measure-batches 20 \
		--output outputs/benchmarks/pi05_trt_int8_suffix_graph_5w20m.json

eval-trt-int8:
	PYTHONPATH=$(PYTHONPATH) PI05_TRT_SUFFIX_GRAPH=1 $(PYTHON) scripts/eval_open_loop_pi05.py \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--backend trt_int8 \
		--output outputs/eval/pi05_trt_int8_suffix_graph_open_loop.json
