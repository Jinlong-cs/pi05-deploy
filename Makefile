bootstrap:
	bash scripts/bootstrap_env.sh

jetson-bootstrap:
	bash scripts/install_jetson_pi05_stack.sh

validate:
	.venv/bin/python scripts/setup_pi05_stack.py --validate

download-metadata:
	.venv/bin/python scripts/download_dataset.py --preset pi05_aloha_public --mode metadata

make-split:
	.venv/bin/python scripts/make_fixed_eval_split.py --preset pi05_aloha_public

download-model:
	.venv/bin/python scripts/download_pi05_assets.py --preset pi05_aloha_public

eval:
	PYTHONPATH=src .venv/bin/python scripts/eval_open_loop_pi05.py --preset pi05_aloha_public --device cuda

benchmark:
	PYTHONPATH=src .venv/bin/python scripts/benchmark_pi05.py --preset pi05_aloha_public --device cuda
