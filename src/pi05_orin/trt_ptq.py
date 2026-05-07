from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


class CalibrationStream(Protocol):
    batch_size: int
    yielded_batches: int

    def next_batch(self, names: list[str]) -> dict[str, torch.Tensor] | None: ...

    def describe(self) -> dict[str, object]: ...


def list_capture_dirs(capture_root: Path) -> list[Path]:
    return sorted(path for path in capture_root.iterdir() if path.is_dir())


def _load_tensor_payload(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict payload at {path}, got {type(payload).__name__}.")
    return payload


@dataclass
class PrefixEmbedCalibrationStream:
    capture_root: Path

    def __post_init__(self) -> None:
        self.sample_dirs = list_capture_dirs(self.capture_root)
        self.index = 0
        self.batch_size = 1
        self.yielded_batches = 0

    def next_batch(self, names: list[str]) -> dict[str, torch.Tensor] | None:
        if self.index >= len(self.sample_dirs):
            return None
        sample_dir = self.sample_dirs[self.index]
        self.index += 1
        inputs_payload = _load_tensor_payload(sample_dir / "inputs.pt")
        self.yielded_batches += 1
        return {name: inputs_payload[name].contiguous() for name in names}

    def describe(self) -> dict[str, object]:
        return {
            "stage": "prefix_embed",
            "capture_root": str(self.capture_root),
            "num_sample_dirs": len(self.sample_dirs),
            "yielded_batches": self.yielded_batches,
        }


@dataclass
class PrefixLmCalibrationStream:
    capture_root: Path

    def __post_init__(self) -> None:
        self.sample_dirs = list_capture_dirs(self.capture_root)
        self.index = 0
        self.batch_size = 1
        self.yielded_batches = 0

    def next_batch(self, names: list[str]) -> dict[str, torch.Tensor] | None:
        if self.index >= len(self.sample_dirs):
            return None
        sample_dir = self.sample_dirs[self.index]
        self.index += 1
        prefix_payload = _load_tensor_payload(sample_dir / "prefix.pt")
        self.yielded_batches += 1
        return {name: prefix_payload[name].contiguous() for name in names}

    def describe(self) -> dict[str, object]:
        return {
            "stage": "prefix_lm",
            "capture_root": str(self.capture_root),
            "num_sample_dirs": len(self.sample_dirs),
            "yielded_batches": self.yielded_batches,
        }


@dataclass
class SuffixStepCalibrationStream:
    capture_root: Path

    def __post_init__(self) -> None:
        self.sample_dirs = list_capture_dirs(self.capture_root)
        self.sample_index = 0
        self.step_index = 0
        self.current_prefix: dict[str, torch.Tensor] | None = None
        self.current_steps: list[dict[str, object]] = []
        self.current_dt: torch.Tensor | None = None
        self.batch_size = 1
        self.yielded_batches = 0

    def _load_current_sample(self) -> bool:
        if self.sample_index >= len(self.sample_dirs):
            return False
        sample_dir = self.sample_dirs[self.sample_index]
        self.current_prefix = _load_tensor_payload(sample_dir / "prefix.pt")
        suffix_loop_path = sample_dir / "suffix_loop.pt"
        if suffix_loop_path.exists():
            payload = torch.load(suffix_loop_path, map_location="cpu")
            self.current_steps = list(payload["steps"])
            dt_value = payload.get("dt")
            if dt_value is None:
                dt_value = -1.0 / max(1, len(self.current_steps))
            self.current_dt = torch.tensor([float(dt_value)], dtype=torch.float32)
        else:
            suffix_payload = _load_tensor_payload(sample_dir / "suffix_step.pt")
            self.current_steps = [
                {
                    "step_index": 0,
                    "timestep": suffix_payload["timestep"],
                    "x_t": suffix_payload["x_t"],
                }
            ]
            dt_value = suffix_payload.get("dt")
            if isinstance(dt_value, torch.Tensor):
                self.current_dt = dt_value.reshape(-1).to(dtype=torch.float32)
            elif dt_value is not None:
                self.current_dt = torch.tensor([float(dt_value)], dtype=torch.float32)
            else:
                self.current_dt = torch.tensor([-1.0], dtype=torch.float32)
        self.step_index = 0
        return True

    def next_batch(self, names: list[str]) -> dict[str, torch.Tensor] | None:
        while True:
            if self.current_prefix is None or self.step_index >= len(self.current_steps):
                if not self._load_current_sample():
                    return None
            if self.step_index >= len(self.current_steps):
                self.sample_index += 1
                self.current_prefix = None
                continue

            assert self.current_prefix is not None
            step_payload = self.current_steps[self.step_index]
            self.step_index += 1
            if self.step_index >= len(self.current_steps):
                self.sample_index += 1
                next_prefix = None
            else:
                next_prefix = self.current_prefix

            batch: dict[str, torch.Tensor] = {}
            for name in names:
                if name == "x_t":
                    batch[name] = step_payload["x_t"].contiguous()  # type: ignore[index]
                elif name == "timestep":
                    batch[name] = step_payload["timestep"].contiguous()  # type: ignore[index]
                elif name == "dt":
                    if self.current_dt is None:
                        raise RuntimeError("Suffix step calibration stream is missing dt.")
                    batch[name] = self.current_dt.contiguous()
                else:
                    batch[name] = self.current_prefix[name].contiguous()

            self.current_prefix = next_prefix
            self.yielded_batches += 1
            return batch

    def describe(self) -> dict[str, object]:
        num_suffix_steps = 0
        for sample_dir in self.sample_dirs:
            suffix_loop_path = sample_dir / "suffix_loop.pt"
            if suffix_loop_path.exists():
                payload = torch.load(suffix_loop_path, map_location="cpu")
                num_suffix_steps += len(payload["steps"])
            else:
                num_suffix_steps += 1
        return {
            "stage": "suffix_step",
            "capture_root": str(self.capture_root),
            "num_sample_dirs": len(self.sample_dirs),
            "num_suffix_steps": num_suffix_steps,
            "yielded_batches": self.yielded_batches,
        }


@dataclass
class SuffixUnrolledCalibrationStream:
    capture_root: Path
    num_inference_steps: int = 10

    def __post_init__(self) -> None:
        self.sample_dirs = list_capture_dirs(self.capture_root)
        self.index = 0
        self.batch_size = 1
        self.yielded_batches = 0

    def next_batch(self, names: list[str]) -> dict[str, torch.Tensor] | None:
        if self.index >= len(self.sample_dirs):
            return None

        sample_dir = self.sample_dirs[self.index]
        self.index += 1
        prefix_payload = _load_tensor_payload(sample_dir / "prefix.pt")
        suffix_loop_payload = torch.load(sample_dir / "suffix_loop.pt", map_location="cpu")
        steps = list(suffix_loop_payload["steps"])
        if len(steps) < self.num_inference_steps:
            raise ValueError(
                f"Capture at {sample_dir} only has {len(steps)} steps, expected at least {self.num_inference_steps}."
            )
        steps = steps[: self.num_inference_steps]
        dt_value = float(suffix_loop_payload.get("dt", -1.0 / max(1, len(steps))))

        batch: dict[str, torch.Tensor] = {}
        for name in names:
            if name == "x_t0":
                batch[name] = steps[0]["x_t"].contiguous()
            elif name == "timesteps":
                batch[name] = torch.cat(
                    [step["timestep"].reshape(-1, 1).contiguous() for step in steps],
                    dim=1,
                ).contiguous()
            elif name == "dt":
                batch[name] = torch.tensor([dt_value], dtype=torch.float32)
            else:
                batch[name] = prefix_payload[name].contiguous()

        self.yielded_batches += 1
        return batch

    def describe(self) -> dict[str, object]:
        num_suffix_steps = 0
        for sample_dir in self.sample_dirs:
            suffix_loop_payload = torch.load(sample_dir / "suffix_loop.pt", map_location="cpu")
            num_suffix_steps += min(self.num_inference_steps, len(suffix_loop_payload["steps"]))
        return {
            "stage": "suffix_unrolled",
            "capture_root": str(self.capture_root),
            "num_sample_dirs": len(self.sample_dirs),
            "num_suffix_steps": num_suffix_steps,
            "num_inference_steps": self.num_inference_steps,
            "yielded_batches": self.yielded_batches,
        }


class _TorchCalibratorState:
    def __init__(
        self,
        *,
        input_names: list[str],
        stream: CalibrationStream,
        cache_file: Path,
        log_every: int = 20,
    ) -> None:
        super().__init__()
        self.input_names = input_names
        self.stream = stream
        self.cache_file = cache_file
        self.log_every = log_every
        self.device_buffers: dict[str, torch.Tensor] = {}

    def get_batch_size(self) -> int:
        return self.stream.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        batch = self.stream.next_batch(names)
        if batch is None:
            print(f"Calibration finished after {self.stream.yielded_batches} batches.")
            return None

        bindings: list[int] = []
        for name in names:
            tensor = batch[name]
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            device_tensor = self.device_buffers.get(name)
            if (
                device_tensor is None
                or tuple(device_tensor.shape) != tuple(tensor.shape)
                or device_tensor.dtype != tensor.dtype
            ):
                device_tensor = torch.empty_like(tensor, device="cuda")
                self.device_buffers[name] = device_tensor

            device_tensor.copy_(tensor, non_blocking=False)
            bindings.append(int(device_tensor.data_ptr()))

        if self.log_every > 0 and self.stream.yielded_batches % self.log_every == 0:
            print(f"Calibration batches consumed: {self.stream.yielded_batches}")
        return bindings

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_file.exists():
            print(f"Reusing calibration cache from {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)
        print(f"Wrote calibration cache to {self.cache_file}")


def make_calibrator(
    *,
    trt: object,
    algorithm: str,
    input_names: list[str],
    stream: CalibrationStream,
    cache_file: Path,
    log_every: int = 20,
) -> object:
    if algorithm == "entropy2":
        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, **kwargs: object) -> None:
                trt.IInt8EntropyCalibrator2.__init__(self)
                self._state = _TorchCalibratorState(**kwargs)

            def get_batch_size(self) -> int:
                return self._state.get_batch_size()

            def get_batch(self, names: list[str]) -> list[int] | None:
                return self._state.get_batch(names)

            def read_calibration_cache(self) -> bytes | None:
                return self._state.read_calibration_cache()

            def write_calibration_cache(self, cache: bytes) -> None:
                self._state.write_calibration_cache(cache)
    elif algorithm == "minmax":
        class Calibrator(trt.IInt8MinMaxCalibrator):
            def __init__(self, **kwargs: object) -> None:
                trt.IInt8MinMaxCalibrator.__init__(self)
                self._state = _TorchCalibratorState(**kwargs)

            def get_batch_size(self) -> int:
                return self._state.get_batch_size()

            def get_batch(self, names: list[str]) -> list[int] | None:
                return self._state.get_batch(names)

            def read_calibration_cache(self) -> bytes | None:
                return self._state.read_calibration_cache()

            def write_calibration_cache(self, cache: bytes) -> None:
                self._state.write_calibration_cache(cache)
    else:
        raise ValueError(f"Unsupported calibration algorithm: {algorithm}")

    return Calibrator(
        input_names=input_names,
        stream=stream,
        cache_file=cache_file,
        log_every=log_every,
    )


def make_calibration_stream(stage: str, capture_root: Path) -> CalibrationStream:
    if stage == "prefix_embed":
        return PrefixEmbedCalibrationStream(capture_root=capture_root)
    if stage == "prefix_lm":
        return PrefixLmCalibrationStream(capture_root=capture_root)
    if stage == "suffix_step":
        return SuffixStepCalibrationStream(capture_root=capture_root)
    if stage == "suffix_unrolled":
        return SuffixUnrolledCalibrationStream(capture_root=capture_root)
    raise ValueError(f"Unsupported PTQ stage: {stage}")


def write_ptq_summary(path: Path, *, stage: str, algorithm: str, stream: CalibrationStream, input_names: list[str]) -> None:
    summary = {
        "stage": stage,
        "algorithm": algorithm,
        "input_names": input_names,
        **stream.describe(),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
