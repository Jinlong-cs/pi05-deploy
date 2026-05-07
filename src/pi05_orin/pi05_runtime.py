from __future__ import annotations

import copy
import logging
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import (
    Pi05PrepareStateTokenizerProcessorStep,
    make_pi05_pre_post_processors,
)
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    ACTION,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    REWARD,
)

from pi05_orin.paths import dataset_root

LOGGER = logging.getLogger(__name__)
DEFAULT_PI05_TOKENIZER = "google/paligemma-3b-pt-224"
PUBLIC_TOKENIZER_FALLBACK = "t5-small"


@dataclass
class PI05RuntimeBundle:
    config: PI05Config
    dataset: LeRobotDataset
    policy: torch.nn.Module
    preprocessor: Any


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pi05_config(model_ref: str | Path, *, revision: str = "main") -> PI05Config:
    # `PI05Config.from_pretrained()` in lerobot 0.5.1 chokes on the serialized `type`
    # field because the initial draccus parse is run against the concrete subclass.
    config = PreTrainedConfig.from_pretrained(str(model_ref), revision=revision)
    if not isinstance(config, PI05Config):
        raise TypeError(f"Expected PI05Config, got {type(config).__name__}")
    return config


class StableHashTokenizer:
    """Minimal tokenizer fallback to keep the AGX benchmark path runnable without gated HF assets."""

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
    ) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def _encode(self, text: str, *, max_length: int, padding_side: str) -> tuple[list[int], list[int]]:
        token_ids = [self.bos_token_id]
        for token in text.strip().split():
            hashed = zlib.crc32(token.encode("utf-8")) % max(1, self.vocab_size - 16)
            token_ids.append(16 + hashed)
        token_ids.append(self.eos_token_id)
        token_ids = token_ids[:max_length]
        attention_mask = [1] * len(token_ids)
        pad_length = max(0, max_length - len(token_ids))
        if padding_side == "left":
            token_ids = [self.pad_token_id] * pad_length + token_ids
            attention_mask = [0] * pad_length + attention_mask
        else:
            token_ids = token_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        return token_ids, attention_mask

    def __call__(
        self,
        text: str | list[str],
        *,
        max_length: int,
        truncation: bool = True,
        padding: str = "max_length",
        padding_side: str = "right",
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del truncation, padding
        if return_tensors != "pt":
            raise ValueError("StableHashTokenizer only supports return_tensors='pt'.")

        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self._encode(sample, max_length=max_length, padding_side=padding_side) for sample in texts]
        input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_tokenizer_fallback(tokenizer_name: str) -> Any:
    try:
        from transformers import AutoTokenizer

        LOGGER.warning("Falling back to tokenizer '%s' for pi0.5 preprocessing.", tokenizer_name)
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:
        LOGGER.warning(
            "Failed to load tokenizer fallback '%s' (%s). Using StableHashTokenizer instead.",
            tokenizer_name,
            exc,
        )
        return StableHashTokenizer()


def make_pi05_pre_post_processors_with_tokenizer(
    *,
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    tokenizer: Any,
) -> tuple[Any, Any]:
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer=tokenizer,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def load_pi05_processors(
    *,
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
) -> tuple[Any, Any]:
    explicit_tokenizer = os.environ.get("PI05_TOKENIZER_NAME")
    if explicit_tokenizer:
        tokenizer = load_tokenizer_fallback(explicit_tokenizer)
        return make_pi05_pre_post_processors_with_tokenizer(
            config=config,
            dataset_stats=dataset_stats,
            tokenizer=tokenizer,
        )

    try:
        return make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    except Exception as exc:
        error_text = str(exc)
        if DEFAULT_PI05_TOKENIZER not in error_text and "gated repo" not in error_text.lower():
            raise

        LOGGER.warning(
            "Falling back from gated tokenizer '%s' after preprocessing init failed: %s",
            DEFAULT_PI05_TOKENIZER,
            exc,
        )
        tokenizer = load_tokenizer_fallback(PUBLIC_TOKENIZER_FALLBACK)
        return make_pi05_pre_post_processors_with_tokenizer(
            config=config,
            dataset_stats=dataset_stats,
            tokenizer=tokenizer,
        )


def adapt_normalization_mapping(
    config: PI05Config,
    *,
    ds_meta: LeRobotDatasetMetadata,
) -> dict[str, NormalizationMode]:
    mapping = dict(config.normalization_mapping)
    quantile_keys = {"q01", "q99"}
    feature_type_to_stats_key = {
        "STATE": "observation.state",
        "ACTION": "action",
    }

    for feature_type_name, stats_key in feature_type_to_stats_key.items():
        mode = mapping.get(feature_type_name)
        feature_stats = ds_meta.stats.get(stats_key, {}) if ds_meta.stats is not None else {}
        if mode is NormalizationMode.QUANTILES and not quantile_keys.issubset(feature_stats):
            LOGGER.warning(
                "Dataset stats for '%s' are missing %s. Falling back from QUANTILES to MIN_MAX normalization.",
                stats_key,
                sorted(quantile_keys),
            )
            mapping[feature_type_name] = NormalizationMode.MIN_MAX

    return mapping


def adapt_config_to_dataset(
    config: PI05Config,
    *,
    ds_meta: LeRobotDatasetMetadata,
    device: str,
    model_ref: str | Path,
    compile_model: bool = False,
) -> PI05Config:
    cfg = copy.deepcopy(config)
    cfg.device = device
    cfg.pretrained_path = str(model_ref)
    cfg.normalization_mapping = adapt_normalization_mapping(cfg, ds_meta=ds_meta)

    features = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {
        key: feature
        for key, feature in features.items()
        if key == "action" or key.startswith("action")
    }
    cfg.input_features = {key: feature for key, feature in features.items() if key not in cfg.output_features}

    if hasattr(cfg, "action_feature_names"):
        action_names = ds_meta.features.get("action", {}).get("names")
        if action_names is not None:
            cfg.action_feature_names = list(action_names)

    if hasattr(cfg, "compile_model"):
        cfg.compile_model = compile_model

    return cfg


def override_inference_steps(config: PI05Config, num_inference_steps: int | None) -> PI05Config:
    if num_inference_steps is None:
        return config
    cfg = copy.deepcopy(config)
    cfg.num_inference_steps = num_inference_steps
    return cfg


def resolve_delta_timestamps(
    config: PI05Config,
    ds_meta: LeRobotDatasetMetadata,
) -> dict[str, list[float]] | None:
    delta_timestamps: dict[str, list[float]] = {}
    for key, feature in ds_meta.features.items():
        feature_type = feature.get("type")
        if key == REWARD and config.reward_delta_indices is not None:
            delta_timestamps[key] = [index / ds_meta.fps for index in config.reward_delta_indices]
        elif key == ACTION and config.action_delta_indices is not None:
            delta_timestamps[key] = [index / ds_meta.fps for index in config.action_delta_indices]
        elif (
            isinstance(feature_type, str)
            and key.startswith(OBS_PREFIX)
            and config.observation_delta_indices is not None
        ):
            delta_timestamps[key] = [index / ds_meta.fps for index in config.observation_delta_indices]
        elif feature_type is FeatureType.ACTION and config.action_delta_indices is not None:
            delta_timestamps[key] = [index / ds_meta.fps for index in config.action_delta_indices]
    return delta_timestamps or None


def make_pi05_dataset(
    *,
    config: PI05Config,
    root: Path | None = None,
    episodes: list[int] | None = None,
    revision: str = "main",
    video_backend: str = "pyav",
) -> LeRobotDataset:
    repo_id = config.repo_id
    root = root or dataset_root(repo_id)
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    delta_timestamps = resolve_delta_timestamps(config, meta)
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        revision=revision,
        download_videos=False,
        video_backend=video_backend,
    )


def load_pi05_runtime(
    *,
    model_ref: str | Path,
    repo_id: str,
    device: str,
    episodes: list[int],
    root: Path | None = None,
    revision: str = "main",
    compile_model: bool = False,
    dtype_override: str | None = None,
    num_inference_steps: int | None = None,
) -> PI05RuntimeBundle:
    model_ref = str(model_ref)
    raw_config = load_pi05_config(model_ref, revision=revision)
    if raw_config.repo_id != repo_id:
        raw_config.repo_id = repo_id

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root or dataset_root(repo_id), revision=revision)
    config = adapt_config_to_dataset(
        raw_config,
        ds_meta=ds_meta,
        device=device,
        model_ref=model_ref,
        compile_model=compile_model,
    )
    config = override_inference_steps(config, num_inference_steps)
    if dtype_override is not None:
        config.dtype = dtype_override

    dataset = make_pi05_dataset(
        config=config,
        root=root,
        episodes=episodes,
        revision=revision,
    )
    policy = PI05Policy.from_pretrained(pretrained_name_or_path=model_ref, config=config)
    preprocessor, _ = load_pi05_processors(config=config, dataset_stats=dataset.meta.stats)
    policy.eval()
    return PI05RuntimeBundle(
        config=config,
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
    )
