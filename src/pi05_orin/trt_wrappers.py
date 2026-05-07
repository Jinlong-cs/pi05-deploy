from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE


def prefix_cache_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for layer_idx in range(num_layers):
        names.append(f"cache_key_{layer_idx:02d}")
        names.append(f"cache_value_{layer_idx:02d}")
    return names


def flatten_dynamic_cache(past_key_values: object) -> tuple[Tensor, ...]:
    if hasattr(past_key_values, "layers"):
        layers = getattr(past_key_values, "layers")
        flat: list[Tensor] = []
        for layer in layers:
            flat.append(layer.keys)
            flat.append(layer.values)
        return tuple(flat)

    flat = []
    for layer in past_key_values:  # type: ignore[assignment]
        flat.extend(layer[:2])
    return tuple(flat)


def legacy_cache_from_flat(flat_cache: tuple[Tensor, ...], num_layers: int) -> tuple[tuple[Tensor, Tensor], ...]:
    if len(flat_cache) != num_layers * 2:
        raise ValueError(f"Expected {num_layers * 2} cache tensors, got {len(flat_cache)}.")
    return tuple((flat_cache[2 * idx], flat_cache[2 * idx + 1]) for idx in range(num_layers))


def dynamic_cache_from_flat(
    flat_cache: tuple[Tensor, ...],
    num_layers: int,
    *,
    config: object | None = None,
) -> object:
    legacy_cache = legacy_cache_from_flat(flat_cache, num_layers)
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache(legacy_cache, config=config)
    except Exception:
        return legacy_cache


def save_io_summary(path: Path, *, inputs: dict[str, Tensor], outputs: dict[str, Tensor]) -> None:
    import json

    summary = {
        "inputs": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in inputs.items()
        },
        "outputs": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in outputs.items()
        },
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def make_att_2d_masks_trt(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class Pi05PrefixEmbedWrapper(nn.Module):
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.image_key = next(iter(policy.config.image_features))

    def forward(self, image: Tensor, tokens: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = {
            self.image_key: image,
        }
        images, img_masks = self.policy._preprocess_images(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
            images,
            img_masks,
            tokens,
            attention_mask,
        )
        prefix_att_2d_masks = make_att_2d_masks_trt(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(dtype=torch.int64), dim=1) - 1
        prefix_att_2d_masks_4d = torch.where(
            prefix_att_2d_masks[:, None, :, :],
            torch.tensor(0.0, device=prefix_embs.device, dtype=prefix_embs.dtype),
            torch.tensor(OPENPI_ATTENTION_MASK_VALUE, device=prefix_embs.device, dtype=prefix_embs.dtype),
        )
        return prefix_embs, prefix_pad_masks, prefix_position_ids, prefix_att_2d_masks_4d


class Pi05PrefixLmWrapper(nn.Module):
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.num_layers = policy.model.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers

    def forward(
        self,
        prefix_embs: Tensor,
        prefix_position_ids: Tensor,
        prefix_attention_mask_4d: Tensor,
    ) -> tuple[Tensor, ...]:
        self.policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_attention_mask_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return flatten_dynamic_cache(past_key_values)


class Pi05SuffixStepWrapper(nn.Module):
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.num_layers = policy.model.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers

    def forward(
        self,
        prefix_pad_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        dt: Tensor,
        *flat_cache: Tensor,
    ) -> Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.policy.model.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks_trt(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks.to(dtype=torch.int64), dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(dtype=torch.int64), dim=1) - 1
        full_att_2d_masks_4d = self.policy.model._prepare_attention_masks_4d(full_att_2d_masks)

        self.policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.policy.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=dynamic_cache_from_flat(
                flat_cache,
                self.num_layers,
                config=self.policy.model.paligemma_with_expert.gemma_expert.model.config,
            ),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.policy.config.chunk_size :].to(dtype=torch.float32)
        v_t = self.policy.model.action_out_proj(suffix_out)
        return x_t + dt.reshape(1, 1, 1).to(dtype=v_t.dtype, device=v_t.device) * v_t


class Pi05SuffixUnrolledWrapper(nn.Module):
    def __init__(self, policy: nn.Module, *, num_inference_steps: int) -> None:
        super().__init__()
        self.step = Pi05SuffixStepWrapper(policy)
        self.num_inference_steps = num_inference_steps

    def forward(
        self,
        prefix_pad_masks: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
        dt: Tensor,
        *flat_cache: Tensor,
    ) -> Tensor:
        if timesteps.ndim != 2:
            raise ValueError(f"Expected timesteps to have rank 2, got {timesteps.ndim}.")
        if timesteps.shape[1] != self.num_inference_steps:
            raise ValueError(
                f"Expected timesteps.shape[1] == {self.num_inference_steps}, got {timesteps.shape[1]}."
            )

        x_next = x_t
        for step_idx in range(self.num_inference_steps):
            x_next = self.step(
                prefix_pad_masks,
                x_next,
                timesteps[:, step_idx],
                dt,
                *flat_cache,
            )
        return x_next
