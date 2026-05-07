#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JETSON_TORCH_INDEX_URL="${JETSON_TORCH_INDEX_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126/+simple}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.5.1}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}" --system-site-packages
fi

"${VENV_DIR}/bin/python" -m ensurepip --upgrade
"${VENV_DIR}/bin/python" -m pip install --upgrade pip 'setuptools>=71,<81' wheel

if [[ -f /etc/nv_tegra_release ]] && command -v apt-get >/dev/null 2>&1; then
  if ! dpkg -s libcudnn9-cuda-12 >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    if ! apt-get update || ! apt-get install -y libcudnn9-cuda-12; then
      echo "warning: failed to install libcudnn9-cuda-12 via apt; continuing with wheel-based validation" >&2
    fi
  fi
fi

"${VENV_DIR}/bin/pip" install \
  --index-url "${JETSON_TORCH_INDEX_URL}" \
  --extra-index-url "${PYPI_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

"${VENV_DIR}/bin/pip" install \
  --index-url "${PYPI_INDEX_URL}" \
  'numpy>=2,<2.3' \
  'datasets>=4,<5' \
  'deepdiff>=7,<9' \
  'draccus==0.10.0' \
  'einops>=0.8,<0.9' \
  'fsspec>=2024.6.1' \
  'gymnasium>=1.1.1,<2' \
  'huggingface-hub>=1,<2' \
  'imageio[ffmpeg]>=2.34,<3' \
  'jsonlines>=4,<5' \
  'packaging>=24.2,<26' \
  'pandas>=2.2,<2.4' \
  'Pillow>=10,<13' \
  'pyarrow>=16,<22' \
  'pyserial>=3.5,<4' \
  'safetensors>=0.4.3,<1' \
  'scipy>=1.14,<2' \
  'sentencepiece>=0.2,<0.3' \
  'termcolor>=2.4,<4' \
  'tqdm>=4.66,<5' \
  'transformers==5.3.0' \
  'av>=15,<16'

"${VENV_DIR}/bin/pip" install \
  --index-url "${PYPI_INDEX_URL}" \
  --ignore-requires-python \
  --no-deps \
  "lerobot==${LEROBOT_VERSION}"

"${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

import pathlib
import site
import sys


def first_existing_site_packages() -> pathlib.Path:
    candidates = [pathlib.Path(p) for p in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(pathlib.Path(user_site))
    for candidate in candidates:
        if (candidate / "lerobot").exists():
            return candidate
    raise SystemExit("Could not locate installed lerobot package in site-packages.")


def replace_once(text: str, old: str, new: str, path: pathlib.Path) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"Expected snippet not found in {path}: {old!r}")
    return text.replace(old, new, 1)


site_packages = first_existing_site_packages()
root = site_packages / "lerobot"
policy_init = root / "policies" / "__init__.py"

policy_init.write_text(
    '"""Lightweight policy package init for Jetson py310 compatibility."""\n\n__all__ = []\n',
    encoding="utf-8",
)
print(f"patched {policy_init}")

patches: dict[pathlib.Path, list[tuple[str, str]]] = {
    root / "datasets" / "streaming_dataset.py": [
        (
            "from collections.abc import Callable, Generator, Iterable, Iterator\nfrom pathlib import Path\n",
            "from collections.abc import Callable, Generator, Iterable, Iterator\nfrom pathlib import Path\nfrom typing import Generic, TypeVar\n",
        ),
        (
            "class Backtrackable[T]:\n",
            'T = TypeVar("T")\n\n\nclass Backtrackable(Generic[T]):\n',
        ),
    ],
    root / "motors" / "motors_bus.py": [
        (
            "type NameOrID = str | int\ntype Value = int | float\n",
            "NameOrID = str | int\nValue = int | float\n",
        ),
    ],
    root / "processor" / "pipeline.py": [
        (
            "from typing import Any, TypedDict, TypeVar, cast\n",
            "from typing import Any, Generic, TypedDict, TypeVar, cast\n",
        ),
        (
            "class DataProcessorPipeline[TInput, TOutput](HubMixin):\n",
            "class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):\n",
        ),
    ],
    root / "utils" / "io_utils.py": [
        (
            "from pathlib import Path\n",
            "from pathlib import Path\nfrom typing import TypeVar\n",
        ),
        (
            'JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]\n\n\ndef write_video(video_path, stacked_frames, fps):\n',
            'JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]\nT = TypeVar("T", bound=JsonLike)\n\n\ndef write_video(video_path, stacked_frames, fps):\n',
        ),
        (
            "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:\n",
            "def deserialize_json_into_object(fpath: Path, obj: T) -> T:\n",
        ),
    ],
    root / "policies" / "pretrained.py": [
        (
            "import abc\n",
            "from __future__ import annotations\n\nimport abc\n",
        ),
        (
            "from typing import TypedDict, TypeVar\nfrom typing_extensions import Unpack\n",
            "from typing import TYPE_CHECKING, TypedDict, TypeVar\nfrom typing_extensions import Unpack\n",
        ),
        (
            "from typing import TypedDict, TypeVar, Unpack\n",
            "from typing import TYPE_CHECKING, TypedDict, TypeVar\nfrom typing_extensions import Unpack\n",
        ),
        (
            "from lerobot.configs.train import TrainPipelineConfig\n",
            "if TYPE_CHECKING:\n    from lerobot.configs.train import TrainPipelineConfig\n",
        ),
    ],
    root / "policies" / "pi05" / "modeling_pi05.py": [
        (
            "from typing import TYPE_CHECKING, Literal, TypedDict, Unpack\n",
            "from typing import TYPE_CHECKING, Literal, TypedDict\nfrom typing_extensions import Unpack\n",
        ),
    ],
}

for path, replacements in patches.items():
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        text = replace_once(text, old, new, path)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")

print(f"Patched lerobot {root} for Python {sys.version.split()[0]} compatibility.")
PY

"${VENV_DIR}/bin/python" -m compileall -q "${VENV_DIR}/lib"
"${VENV_DIR}/bin/python" scripts/setup_pi05_stack.py --validate
