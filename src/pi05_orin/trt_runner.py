from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from pi05_orin.paths import dataset_root, trt_engine_dir
from pi05_orin.pi05_runtime import (
    PI05RuntimeBundle,
    adapt_config_to_dataset,
    load_pi05_config,
    load_pi05_processors,
    make_pi05_dataset,
    override_inference_steps,
)


LOGGER = logging.getLogger(__name__)


def _load_tensorrt() -> Any:
    import tensorrt as trt

    return trt


def _trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    return mapping[dtype]


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def resolve_stage_engine_path(engine_root: Path, *, stage: str, precision: str) -> Path:
    stage_dir = engine_root / precision
    candidates = [
        stage_dir / f"{stage}.engine",
        stage_dir / f"{stage}.plan",
        engine_root / f"{stage}_{precision}.engine",
        engine_root / f"{stage}_{precision}.plan",
        engine_root / f"{stage}.engine",
        engine_root / f"{stage}.plan",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate TensorRT engine for stage '{stage}' with precision '{precision}'. "
        f"Searched:\n{searched}"
    )


class TrtEngineRunner:
    def __init__(self, engine_path: Path) -> None:
        trt = _load_tensorrt()
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine_bytes = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {engine_path}")

        self.trt = trt
        self.engine = engine
        self.context = context
        self.tensor_names = [engine.get_tensor_name(idx) for idx in range(engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

    def _normalize_input_tensors(self, feeds: dict[str, Tensor]) -> dict[str, Tensor]:
        normalized: dict[str, Tensor] = {}
        for name in self.input_names:
            tensor = feeds[name]
            if tensor.device.type != "cuda":
                tensor = tensor.to(device="cuda")
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self.context.set_input_shape(name, tuple(tensor.shape))
            normalized[name] = tensor
        return normalized

    def make_output_tensors(self, feeds: dict[str, Tensor]) -> dict[str, Tensor]:
        self._normalize_input_tensors(feeds)
        outputs: dict[str, Tensor] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
        return outputs

    def ensure_output_tensors(
        self,
        feeds: dict[str, Tensor],
        outputs: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        if outputs is None:
            return self.make_output_tensors(feeds)

        self._normalize_input_tensors(feeds)
        for name in self.output_names:
            expected_shape = tuple(self.context.get_tensor_shape(name))
            expected_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            output = outputs.get(name)
            if output is None:
                return self.make_output_tensors(feeds)
            if tuple(output.shape) != expected_shape or output.dtype != expected_dtype:
                return self.make_output_tensors(feeds)
            if output.device.type != "cuda" or not output.is_contiguous():
                return self.make_output_tensors(feeds)
        return outputs

    def infer_into(self, feeds: dict[str, Tensor], outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        normalized_inputs = self._normalize_input_tensors(feeds)
        stream = torch.cuda.current_stream().cuda_stream

        for name, tensor in normalized_inputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        for name in self.output_names:
            output = outputs[name]
            expected_shape = tuple(self.context.get_tensor_shape(name))
            expected_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            if output.device.type != "cuda":
                raise ValueError(f"Output tensor for '{name}' must live on CUDA, got {output.device}.")
            if tuple(output.shape) != expected_shape:
                raise ValueError(f"Output tensor for '{name}' has shape {tuple(output.shape)}, expected {expected_shape}.")
            if output.dtype != expected_dtype:
                raise ValueError(f"Output tensor for '{name}' has dtype {output.dtype}, expected {expected_dtype}.")
            if not output.is_contiguous():
                raise ValueError(f"Output tensor for '{name}' must be contiguous.")
            self.context.set_tensor_address(name, output.data_ptr())

        ok = self.context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False.")
        return outputs

    def infer(self, feeds: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = self.make_output_tensors(feeds)
        return self.infer_into(feeds, outputs)


class _SuffixCudaGraphLoop:
    def __init__(
        self,
        *,
        engine: TrtEngineRunner,
        cache_names: list[str],
        num_inference_steps: int,
        dt: float,
        enabled: bool,
    ) -> None:
        self.engine = engine
        self.cache_names = cache_names
        self.num_inference_steps = num_inference_steps
        self.dt = dt
        self.enabled = enabled
        self.capture_signature: tuple[object, ...] | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_x_t: Tensor | None = None
        self.static_dt: Tensor | None = None
        self.static_timesteps: list[Tensor] = []
        self.shared_outputs: dict[str, Tensor] | None = None
        self.capture_error: str | None = None

    def _signature(self, *, prefix_pad_masks: Tensor, cache_tensors: list[Tensor], x_t: Tensor) -> tuple[object, ...]:
        cache_signature = tuple(
            (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))
            for tensor in cache_tensors
        )
        return (
            prefix_pad_masks.data_ptr(),
            tuple(prefix_pad_masks.shape),
            str(prefix_pad_masks.dtype),
            cache_signature,
            tuple(x_t.shape),
            str(x_t.dtype),
            x_t.device.index,
        )

    def _make_feed(
        self,
        *,
        prefix_pad_masks: Tensor,
        cache_tensors: list[Tensor],
        step_idx: int,
    ) -> dict[str, Tensor]:
        if self.static_x_t is None:
            raise RuntimeError("Suffix CUDA graph static x_t buffer is not initialized.")
        if self.static_dt is None:
            raise RuntimeError("Suffix CUDA graph static dt buffer is not initialized.")
        feed: dict[str, Tensor] = {
            "prefix_pad_masks": prefix_pad_masks,
            "x_t": self.static_x_t,
            "timestep": self.static_timesteps[step_idx],
            "dt": self.static_dt,
        }
        feed.update({name: tensor for name, tensor in zip(self.cache_names, cache_tensors, strict=True)})
        return feed

    def _capture(self, *, prefix_pad_masks: Tensor, cache_tensors: list[Tensor], x_t: Tensor) -> None:
        self.static_x_t = torch.empty_like(x_t)
        self.static_dt = torch.tensor([self.dt], dtype=torch.float32, device=x_t.device)
        batch_size = x_t.shape[0]
        self.static_timesteps = [
            torch.full((batch_size,), 1.0 + step * self.dt, dtype=torch.float32, device=x_t.device)
            for step in range(self.num_inference_steps)
        ]
        first_feed = self._make_feed(prefix_pad_masks=prefix_pad_masks, cache_tensors=cache_tensors, step_idx=0)
        self.shared_outputs = self.engine.make_output_tensors(first_feed)
        if "x_t_next" in self.shared_outputs:
            self.shared_outputs["x_t_next"] = self.static_x_t
        self.graph = torch.cuda.CUDAGraph()
        self.static_x_t.copy_(x_t)
        with torch.cuda.graph(self.graph):
            for step_idx in range(self.num_inference_steps):
                feed = self._make_feed(
                    prefix_pad_masks=prefix_pad_masks,
                    cache_tensors=cache_tensors,
                    step_idx=step_idx,
                )
                self.engine.infer_into(feed, self.shared_outputs)

    def run(self, *, prefix_pad_masks: Tensor, cache_tensors: list[Tensor], x_t: Tensor) -> Tensor | None:
        if not self.enabled or prefix_pad_masks.device.type != "cuda":
            return None

        signature = self._signature(prefix_pad_masks=prefix_pad_masks, cache_tensors=cache_tensors, x_t=x_t)
        if signature != self.capture_signature:
            self._capture(prefix_pad_masks=prefix_pad_masks, cache_tensors=cache_tensors, x_t=x_t)
            self.capture_signature = signature

        if self.graph is None or self.static_x_t is None:
            return None

        self.static_x_t.copy_(x_t)
        self.graph.replay()
        return self.static_x_t


class TrtPi05Runner(nn.Module):
    def __init__(
        self,
        *,
        config: Any,
        prefix_embed_engine: Path,
        prefix_lm_engine: Path,
        suffix_step_engine: Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.prefix_embed_engine = TrtEngineRunner(prefix_embed_engine)
        self.prefix_lm_engine = TrtEngineRunner(prefix_lm_engine)
        self.suffix_step_engine = TrtEngineRunner(suffix_step_engine)
        self.cache_names = [
            name for name in self.prefix_lm_engine.output_names if name.startswith("cache_")
        ]
        self.image_key = next(iter(config.image_features))
        self.prefix_embed_outputs: dict[str, Tensor] | None = None
        self.prefix_lm_outputs: dict[str, Tensor] | None = None
        self.enable_suffix_cuda_graph = _env_flag("PI05_TRT_SUFFIX_GRAPH", True)
        self.suffix_cuda_graph = _SuffixCudaGraphLoop(
            engine=self.suffix_step_engine,
            cache_names=self.cache_names,
            num_inference_steps=self.config.num_inference_steps,
            dt=-1.0 / self.config.num_inference_steps,
            enabled=self.enable_suffix_cuda_graph,
        )

    def _prepare_prefix_embed_image(self, batch: dict[str, Tensor]) -> Tensor:
        image = batch[self.image_key]
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        if not image.is_contiguous():
            image = image.contiguous()
        return image

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        # The exported prefix engine already includes the model-side image preprocessing path.
        image = self._prepare_prefix_embed_image(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embed_feeds = {
            "image": image,
            "tokens": tokens,
            "attention_mask": attention_mask,
        }
        self.prefix_embed_outputs = self.prefix_embed_engine.ensure_output_tensors(
            prefix_embed_feeds,
            self.prefix_embed_outputs,
        )
        prefix_embed_outputs = self.prefix_embed_engine.infer_into(prefix_embed_feeds, self.prefix_embed_outputs)
        prefix_embs = prefix_embed_outputs["prefix_embs"]
        prefix_pad_masks = prefix_embed_outputs["prefix_pad_masks"]
        prefix_position_ids = prefix_embed_outputs["prefix_position_ids"]
        prefix_attention_mask_4d = prefix_embed_outputs["prefix_attention_mask_4d"]

        prefix_lm_feeds = {
            "prefix_embs": prefix_embs,
            "prefix_position_ids": prefix_position_ids,
            "prefix_attention_mask_4d": prefix_attention_mask_4d,
        }
        self.prefix_lm_outputs = self.prefix_lm_engine.ensure_output_tensors(
            prefix_lm_feeds,
            self.prefix_lm_outputs,
        )
        prefix_lm_outputs = self.prefix_lm_engine.infer_into(prefix_lm_feeds, self.prefix_lm_outputs)
        cache_tensors = [prefix_lm_outputs[name] for name in self.cache_names]

        batch_size = tokens.shape[0]
        x_t = self.sample_noise(
            (batch_size, self.config.chunk_size, self.config.max_action_dim),
            device=tokens.device,
        )
        dt = -1.0 / self.config.num_inference_steps
        dt_tensor = torch.tensor([dt], dtype=torch.float32, device=tokens.device)
        graph_result = self.suffix_cuda_graph.run(
            prefix_pad_masks=prefix_pad_masks,
            cache_tensors=cache_tensors,
            x_t=x_t,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        if graph_result is not None:
            return graph_result[:, :, :original_action_dim].clone()

        for step in range(self.config.num_inference_steps):
            time = 1.0 + step * dt
            timestep = torch.full((batch_size,), time, dtype=torch.float32, device=tokens.device)
            suffix_inputs: dict[str, Tensor] = {
                "prefix_pad_masks": prefix_pad_masks,
                "x_t": x_t,
                "timestep": timestep,
                "dt": dt_tensor,
            }
            suffix_inputs.update({name: tensor for name, tensor in zip(self.cache_names, cache_tensors, strict=True)})
            suffix_outputs = self.suffix_step_engine.infer(suffix_inputs)
            x_t = suffix_outputs["x_t_next"]

        return x_t[:, :, :original_action_dim]


def load_trt_pi05_runtime(
    *,
    model_ref: str | Path,
    repo_id: str,
    device: str,
    episodes: list[int],
    root: Path | None = None,
    revision: str = "main",
    preset_name: str,
    precision: str,
    engine_root: Path | None = None,
    num_inference_steps: int | None = None,
) -> PI05RuntimeBundle:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

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
        compile_model=False,
    )
    config = override_inference_steps(config, num_inference_steps)
    dataset = make_pi05_dataset(
        config=config,
        root=root,
        episodes=episodes,
        revision=revision,
    )
    preprocessor, _ = load_pi05_processors(config=config, dataset_stats=dataset.meta.stats)

    engine_root = engine_root or trt_engine_dir(preset_name)
    runner = TrtPi05Runner(
        config=config,
        prefix_embed_engine=resolve_stage_engine_path(engine_root, stage="prefix_embed", precision=precision),
        prefix_lm_engine=resolve_stage_engine_path(engine_root, stage="prefix_lm", precision=precision),
        suffix_step_engine=resolve_stage_engine_path(engine_root, stage="suffix_step", precision=precision),
    )
    return PI05RuntimeBundle(config=config, dataset=dataset, policy=runner, preprocessor=preprocessor)
