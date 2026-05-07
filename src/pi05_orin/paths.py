import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value).expanduser()


def _prefer_existing(name: str, *candidates: Path) -> Path:
    override = os.environ.get(name)
    if override:
        return Path(override).expanduser()

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_DIR = _env_path("PI05_DATA_DIR", REPO_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = _env_path("PI05_OUTPUTS_DIR", REPO_ROOT / "outputs")
TRT_DIR = _env_path("PI05_TRT_DIR", OUTPUTS_DIR / "trt")
TRT_CAPTURES_DIR = _prefer_existing(
    "PI05_TRT_CAPTURES_DIR",
    OUTPUTS_DIR / "trt_captures",
    TRT_DIR / "captures",
)
TRT_ONNX_DIR = _prefer_existing(
    "PI05_TRT_ONNX_DIR",
    OUTPUTS_DIR / "trt_onnx",
    TRT_DIR / "onnx",
)
TRT_ENGINES_DIR = _prefer_existing(
    "PI05_TRT_ENGINES_DIR",
    OUTPUTS_DIR / "trt_engines",
    TRT_DIR / "engines",
)
TRT_REPORTS_DIR = _prefer_existing(
    "PI05_TRT_REPORTS_DIR",
    OUTPUTS_DIR / "trt_reports",
    TRT_DIR / "reports",
)
CACHE_DIR = _env_path("PI05_CACHE_DIR", REPO_ROOT / ".cache")
THIRD_PARTY_DIR = _env_path("PI05_THIRD_PARTY_DIR", REPO_ROOT / "third_party")
ARTIFACTS_DIR = _env_path("PI05_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def dataset_root(repo_id: str) -> Path:
    return RAW_DIR / repo_slug(repo_id)


def split_dir(repo_id: str) -> Path:
    return SPLITS_DIR / repo_slug(repo_id)


def model_dir(repo_id: str) -> Path:
    return MODELS_DIR / repo_slug(repo_id)


def trt_capture_dir(preset_name: str) -> Path:
    return TRT_CAPTURES_DIR / preset_name


def trt_onnx_dir(preset_name: str) -> Path:
    return TRT_ONNX_DIR / preset_name


def trt_engine_dir(preset_name: str) -> Path:
    return TRT_ENGINES_DIR / preset_name


def trt_report_dir(preset_name: str) -> Path:
    return TRT_REPORTS_DIR / preset_name
