from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wam.utils.logging_config import get_logger

from .visual_history import HistoryAwareWAM, _as_plain_dict
from .mot import MoT
from .frozen_target_signature import frozen_target_feature_spec
from .aed_core import SETFrozenLingBotVisionTarget, SETFrozenVisionStateTarget
from .wan_video_dit import flash_attention

logger = get_logger(__name__)


_DINO_MEMORY_SOURCES = {
    "dino_memory",
    "dinov3_memory",
    "adaptive_dino_memory",
    "multi_frame_dino",
}
_VAE_MEMORY_SOURCES = {
    "vae_memory",
    "adaptive_vae_memory",
    "multi_frame_vae",
}
_HISTORY_MEMORY_ARCHITECTURE_KEYS = (
    "source_family",
    "num_layers",
    "num_queries",
    "input_dim",
    "hidden_dim",
    "text_cross_attention_block_index",
    "text_input_dim",
    "text_projector_kind",
)
_HWEL_PROFILE_FORMAT_VERSION = 1


def _normalize_feature_loss_type(value: Any) -> str:
    normalized = str(value or "normalized_mse").strip().lower()
    aliases = {
        "normalized": "normalized_mse",
        "normalized_delta": "normalized_mse",
        "normalized_delta_mse": "normalized_mse",
        "hwel": "hwel_diagonal",
        "diagonal_hwel": "hwel_diagonal",
        "hwel_batch": "hwel_diagonal_batch",
        "batch_hwel": "hwel_diagonal_batch",
        "online_hwel": "hwel_diagonal_batch",
        "hwel_online": "hwel_diagonal_batch",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "normalized_mse",
        "raw_mse",
        "hwel_diagonal",
        "hwel_diagonal_batch",
    }:
        raise ValueError(
            "`set.feature_loss_type` must be one of "
            "{'normalized_mse', 'raw_mse', 'hwel_diagonal', "
            "'hwel_diagonal_batch'}, got "
            f"{value!r}."
        )
    return normalized


def _load_diagonal_hwel_profile(
    path: str | os.PathLike[str],
    *,
    expected_feature_dim: int,
    expected_token_mode: str,
    expected_target_type: str,
    expected_model_id: str,
    expected_feature_spec: Optional[Mapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], str]:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"Missing diagonal HWEL profile: {profile_path}")
    raw = profile_path.read_bytes()
    payload = json.loads(raw)
    if int(payload.get("format_version", -1)) != _HWEL_PROFILE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported HWEL profile format in {profile_path}: "
            f"{payload.get('format_version')!r}."
        )
    if str(payload.get("profile_type", "")) != "diagonal_hwel":
        raise ValueError(
            f"Expected a diagonal_hwel profile in {profile_path}, got "
            f"{payload.get('profile_type')!r}."
        )
    feature_dim = int(payload.get("feature_dim", -1))
    if feature_dim != int(expected_feature_dim):
        raise ValueError(
            f"HWEL profile feature_dim mismatch: profile={feature_dim}, "
            f"model={expected_feature_dim}."
        )
    token_mode = str(payload.get("token_mode", ""))
    if token_mode != str(expected_token_mode):
        raise ValueError(
            f"HWEL profile token_mode mismatch: profile={token_mode!r}, "
            f"model={expected_token_mode!r}."
        )
    target_type = str(payload.get("target_type", "")).lower()
    expected_target_type = str(expected_target_type).lower()
    if target_type != expected_target_type:
        raise ValueError(
            f"HWEL profile target_type mismatch: profile={target_type!r}, "
            f"model={expected_target_type!r}."
        )
    target_model_id = str(payload.get("target_model_id", ""))
    if target_model_id != str(expected_model_id):
        raise ValueError(
            f"HWEL profile target_model_id mismatch: profile={target_model_id!r}, "
            f"model={expected_model_id!r}."
        )
    if expected_feature_spec is not None:
        profile_feature_spec = payload.get("feature_spec")
        if not isinstance(profile_feature_spec, dict):
            raise ValueError(
                f"HWEL profile has no frozen-target feature_spec: {profile_path}."
            )
        signature_mismatches = {
            key: {"profile": profile_feature_spec.get(key), "model": expected_value}
            for key, expected_value in expected_feature_spec.items()
            if profile_feature_spec.get(key) != expected_value
        }
        if signature_mismatches:
            raise ValueError(
                "HWEL profile frozen-target signature mismatch: "
                f"{signature_mismatches}. Profile={profile_path}."
            )

    span_payload = payload.get("spans")
    if not isinstance(span_payload, dict) or not span_payload:
        raise ValueError(f"HWEL profile has no span statistics: {profile_path}")
    spans = sorted(int(key) for key in span_payload)
    if spans[0] <= 0:
        raise ValueError(f"HWEL profile spans must be positive, got {spans}.")
    table_size = spans[-1] + 1
    inv_second_moment = torch.zeros(table_size, feature_dim, dtype=torch.float32)
    normalizer_z = torch.zeros(table_size, dtype=torch.float32)
    available = torch.zeros(table_size, dtype=torch.bool)
    for span in spans:
        stats = span_payload[str(span)]
        inv = torch.as_tensor(stats.get("inverse_second_moment"), dtype=torch.float32)
        if inv.shape != (feature_dim,):
            raise ValueError(
                f"HWEL inverse_second_moment for span={span} must be [{feature_dim}], "
                f"got {tuple(inv.shape)}."
            )
        z = float(stats.get("normalizer_z", float("nan")))
        if not bool(torch.isfinite(inv).all()) or bool((inv <= 0).any()):
            raise ValueError(f"HWEL inverse_second_moment for span={span} must be finite and positive.")
        if not math.isfinite(z) or z <= 0:
            raise ValueError(f"HWEL normalizer_z for span={span} must be finite and positive, got {z}.")
        inv_second_moment[span] = inv
        normalizer_z[span] = z
        available[span] = True
    return (
        inv_second_moment,
        normalizer_z,
        available,
        payload,
        hashlib.sha256(raw).hexdigest(),
    )


def _history_memory_source_family(source: Any) -> str:
    normalized = str(source or "").lower()
    if normalized in _VAE_MEMORY_SOURCES:
        return "vae_memory"
    if normalized in _DINO_MEMORY_SOURCES:
        return "dino_memory"
    return normalized


def _history_memory_architecture_from_module(
    history_config: dict[str, Any],
    memory: "HistoryVisualMemoryExtractor",
) -> dict[str, Any]:
    projector = memory.text_projector
    if projector is None:
        projector_kind = None
    elif isinstance(projector, nn.Identity):
        projector_kind = "identity"
    elif isinstance(projector, nn.Linear):
        projector_kind = "linear"
    else:
        projector_kind = type(projector).__name__
    return {
        "source_family": _history_memory_source_family(
            history_config.get("visual_token_source")
        ),
        "num_layers": int(memory.num_layers),
        "num_queries": int(memory.num_queries),
        "input_dim": int(memory.input_dim),
        "hidden_dim": int(memory.hidden_dim),
        "text_cross_attention_block_index": memory.text_cross_attention_block_index,
        "text_input_dim": memory.text_input_dim,
        "text_projector_kind": projector_kind,
    }


def _history_memory_architecture_from_checkpoint(
    payload: dict[str, Any],
) -> Optional[dict[str, Any]]:
    metadata = payload.get("history_visual_memory_architecture")
    if metadata is not None:
        metadata = _as_plain_dict(
            metadata,
            "checkpoint.history_visual_memory_architecture",
        )
        return {
            key: metadata.get(key)
            for key in _HISTORY_MEMORY_ARCHITECTURE_KEYS
        }

    state = payload.get("history_visual_memory")
    if not isinstance(state, dict):
        return None
    history_config = _as_plain_dict(
        payload.get("history_config"),
        "checkpoint.history_config",
    )
    memory_config = _as_plain_dict(
        history_config.get("visual_memory"),
        "checkpoint.history_config.visual_memory",
    )
    block_indices = []
    for key in state:
        parts = str(key).split(".")
        if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
            block_indices.append(int(parts[1]))
    inferred_num_layers = max(block_indices) + 1 if block_indices else None
    memory_queries = state.get("memory_queries")
    input_projector_weight = state.get("input_projector.weight")
    text_projector_weight = state.get("text_projector.weight")
    text_projector_keys = [
        key for key in state if str(key).startswith("text_projector.")
    ]
    text_block_index = memory_config.get("text_cross_attention_block_index")
    if text_block_index is not None:
        text_block_index = int(text_block_index)
    if isinstance(text_projector_weight, torch.Tensor):
        text_input_dim = int(text_projector_weight.shape[1])
        text_projector_kind = "linear"
    else:
        text_input_dim = None
        text_projector_kind = "unknown" if text_projector_keys else None
    inferred_hidden_dim = (
        int(memory_queries.shape[2])
        if isinstance(memory_queries, torch.Tensor)
        else None
    )
    return {
        "source_family": _history_memory_source_family(
            history_config.get("visual_token_source")
        ),
        "num_layers": int(
            memory_config.get("num_layers", inferred_num_layers)
        )
        if memory_config.get("num_layers", inferred_num_layers) is not None
        else None,
        "num_queries": (
            int(memory_queries.shape[1])
            if isinstance(memory_queries, torch.Tensor)
            else memory_config.get("num_queries")
        ),
        "input_dim": (
            int(input_projector_weight.shape[1])
            if isinstance(input_projector_weight, torch.Tensor)
            else inferred_hidden_dim
        ),
        "hidden_dim": inferred_hidden_dim,
        "text_cross_attention_block_index": text_block_index,
        "text_input_dim": text_input_dim,
        "text_projector_kind": text_projector_kind,
    }


def _history_memory_architecture_mismatches(
    checkpoint_architecture: Optional[dict[str, Any]],
    runtime_architecture: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "checkpoint": (
                None
                if checkpoint_architecture is None
                else checkpoint_architecture.get(key)
            ),
            "runtime": runtime_architecture.get(key),
        }
        for key in _HISTORY_MEMORY_ARCHITECTURE_KEYS
        if checkpoint_architecture is None
        or checkpoint_architecture.get(key) != runtime_architecture.get(key)
    }


class SerialFeaturePredictionBlock(nn.Module):
    """Decoder-style block that lets DINO patch tokens read action hidden states."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        ffn_dim: int,
        eps: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}.")
        head_dim = self.hidden_dim // self.num_heads
        self.norm_self = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.norm_cross_q = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.norm_cross_kv = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.norm_ffn = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.self_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_o = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_o = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ffn_dim)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(ffn_dim), self.hidden_dim),
        )
        self._attn_head_dim = head_dim

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn_mask is None:
            attn_mask = torch.ones(
                (q.shape[1], k.shape[1]),
                device=q.device,
                dtype=torch.bool,
            )
        else:
            if attn_mask.ndim != 2 or attn_mask.shape != (q.shape[0], k.shape[1]):
                raise ValueError(
                    "Cross-attention mask must be [B,K], got "
                    f"{tuple(attn_mask.shape)} for q={tuple(q.shape)} and k={tuple(k.shape)}."
                )
            attn_mask = attn_mask.to(device=q.device, dtype=torch.bool)[:, None, None, :]
        return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

    def forward(
        self,
        feature_tokens: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.norm_self(feature_tokens)
        self_out = self._attn(self.self_q(x_norm), self.self_k(x_norm), self.self_v(x_norm))
        x = feature_tokens + self.self_o(self_out)

        q = self.cross_q(self.norm_cross_q(x))
        kv = self.norm_cross_kv(action_context)
        cross_out = self._attn(
            q,
            self.cross_k(kv),
            self.cross_v(kv),
            attn_mask=action_context_mask,
        )
        x = x + self.cross_o(cross_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class HistoryVisualMemoryExtractor(nn.Module):
    """Compress variable-length multi-frame visual patches into fixed memory tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_queries: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        max_frames: int = 33,
        eps: float = 1e-6,
        skip_block_indices: Optional[list[int]] = None,
        svd_rank: Optional[int] = None,
        text_input_dim: Optional[int] = None,
        text_cross_attention_block_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.max_frames = int(max_frames)
        self.svd_rank = None if svd_rank is None else int(svd_rank)
        self.text_input_dim = None if text_input_dim is None else int(text_input_dim)
        self.text_cross_attention_block_index = (
            None
            if text_cross_attention_block_index is None
            else int(text_cross_attention_block_index)
        )
        self.skip_block_indices = frozenset(
            int(index) for index in (skip_block_indices or [])
        )
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError(
                f"History visual memory dimensions must be positive, got {self.input_dim} and {self.hidden_dim}."
            )
        if self.num_queries <= 0 or self.num_layers <= 0 or self.max_frames <= 0:
            raise ValueError(
                "History visual memory num_queries, num_layers, and max_frames must be positive, got "
                f"{self.num_queries}, {self.num_layers}, and {self.max_frames}."
            )
        invalid_skip_indices = sorted(
            index
            for index in self.skip_block_indices
            if index < 0 or index >= self.num_layers
        )
        if invalid_skip_indices:
            raise ValueError(
                "History visual memory skip_block_indices must be valid zero-based "
                f"block indices for {self.num_layers} layers, got {invalid_skip_indices}."
            )
        if self.text_cross_attention_block_index is not None:
            if not (
                0 <= self.text_cross_attention_block_index < self.num_layers
            ):
                raise ValueError(
                    "History visual memory text_cross_attention_block_index must be "
                    f"a valid zero-based block index for {self.num_layers} layers, got "
                    f"{self.text_cross_attention_block_index}."
                )
            if self.text_input_dim is None or self.text_input_dim <= 0:
                raise ValueError(
                    "History visual memory text_input_dim must be positive when a "
                    "text cross-attention block is configured, got "
                    f"{self.text_input_dim}."
                )
        max_centered_rank = min(self.num_queries - 1, self.hidden_dim)
        if self.svd_rank is not None and not (
            1 <= self.svd_rank <= max_centered_rank
        ):
            raise ValueError(
                "History visual memory svd_rank must be within the centered slot-rank "
                f"range [1, {max_centered_rank}], got {self.svd_rank}."
            )
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(
                f"History visual memory hidden_dim={self.hidden_dim} must be divisible by num_heads={num_heads}."
            )

        self.input_norm = nn.LayerNorm(self.input_dim, eps=float(eps))
        self.input_projector = (
            nn.Identity()
            if self.input_dim == self.hidden_dim
            else nn.Linear(self.input_dim, self.hidden_dim)
        )
        self.memory_queries = nn.Parameter(torch.zeros(1, self.num_queries, self.hidden_dim))
        self.frame_pos = nn.Parameter(torch.zeros(1, self.max_frames, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.memory_queries, std=0.02)
        nn.init.trunc_normal_(self.frame_pos, std=0.02)
        if self.text_cross_attention_block_index is None:
            self.text_projector = None
        elif self.text_input_dim == self.hidden_dim:
            self.text_projector = nn.Identity()
        else:
            self.text_projector = nn.Linear(self.text_input_dim, self.hidden_dim)
        self._validated_nonempty_text_context = False
        resolved_ffn_dim = int(ffn_dim or self.hidden_dim * 4)
        self.blocks = nn.ModuleList(
            [
                SerialFeaturePredictionBlock(
                    hidden_dim=self.hidden_dim,
                    ffn_dim=resolved_ffn_dim,
                    eps=float(eps),
                    num_heads=int(num_heads),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=float(eps))

    def forward(
        self,
        frame_patch_tokens: torch.Tensor,
        *,
        text_context: Optional[torch.Tensor] = None,
        text_context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if frame_patch_tokens.ndim != 4:
            raise ValueError(
                "`frame_patch_tokens` must be [B,T,P,D], got "
                f"{tuple(frame_patch_tokens.shape)}."
            )
        batch, num_frames, patches_per_frame, input_dim = frame_patch_tokens.shape
        if input_dim != self.input_dim:
            raise ValueError(
                f"History visual token dim must be {self.input_dim}, got {input_dim}."
            )
        if num_frames <= 0 or patches_per_frame <= 0:
            raise ValueError(
                f"History visual memory requires non-empty frames and patches, got T={num_frames}, P={patches_per_frame}."
            )
        if num_frames > self.max_frames:
            raise ValueError(
                f"History visual frames {num_frames} exceed visual_memory.max_frames={self.max_frames}."
            )

        context = self.input_projector(self.input_norm(frame_patch_tokens))
        frame_pos_start = self.max_frames - num_frames
        frame_pos = self.frame_pos[:, frame_pos_start:].to(device=context.device, dtype=context.dtype)
        context = context + frame_pos
        context = context.reshape(batch, num_frames * patches_per_frame, self.hidden_dim)

        projected_text_context = None
        projected_text_mask = None
        text_block_is_active = (
            self.text_cross_attention_block_index is not None
            and self.text_cross_attention_block_index not in self.skip_block_indices
        )
        if text_block_is_active:
            if text_context is None:
                raise ValueError(
                    "History visual memory requires `text_context` because "
                    f"block {self.text_cross_attention_block_index} is configured "
                    "for text cross-attention."
                )
            if text_context.ndim != 3:
                raise ValueError(
                    "`text_context` must be [B,L,D_text], got "
                    f"{tuple(text_context.shape)}."
                )
            if text_context.shape[0] != batch or text_context.shape[2] != self.text_input_dim:
                raise ValueError(
                    "History visual memory text context shape mismatch: expected "
                    f"[{batch},L,{self.text_input_dim}], got {tuple(text_context.shape)}."
                )
            if text_context.shape[1] <= 0:
                raise ValueError("History visual memory requires at least one text token.")
            if text_context_mask is None:
                projected_text_mask = torch.ones(
                    (batch, text_context.shape[1]),
                    device=context.device,
                    dtype=torch.bool,
                )
            else:
                if text_context_mask.ndim != 2 or text_context_mask.shape != text_context.shape[:2]:
                    raise ValueError(
                        "`text_context_mask` must be [B,L] matching text_context, got "
                        f"{tuple(text_context_mask.shape)} vs {tuple(text_context.shape)}."
                    )
                projected_text_mask = text_context_mask.to(
                    device=context.device,
                    dtype=torch.bool,
                )
            # Wan2.2 compatibility keeps zero-padded text embeddings visible by
            # replacing the tokenizer mask with all True. Recover the effective
            # text-only mask before the learnable projector bias can turn those
            # zero rows into non-zero keys and values.
            nonzero_text_mask = (
                text_context.detach().float().abs().amax(dim=-1) > 0
            ).to(device=context.device)
            projected_text_mask = projected_text_mask & nonzero_text_mask
            if not self.training or not self._validated_nonempty_text_context:
                if not bool(projected_text_mask.any(dim=1).all()):
                    raise ValueError(
                        "Each sample must expose at least one valid text token to the "
                        "text cross-attention memory block."
                    )
                # Avoid a host/device synchronization on every training step.
                # Evaluation still validates every prompt.
                self._validated_nonempty_text_context = True
            projected_text_context = self.text_projector(
                text_context.to(device=context.device, dtype=context.dtype)
            )

        memory = self.memory_queries.expand(batch, -1, -1).to(device=context.device, dtype=context.dtype)
        for block_index, block in enumerate(self.blocks):
            if block_index in self.skip_block_indices:
                continue
            if block_index == self.text_cross_attention_block_index:
                block_context = projected_text_context
                block_context_mask = projected_text_mask
            else:
                block_context = context
                block_context_mask = None
            memory = block(
                memory,
                block_context,
                action_context_mask=block_context_mask,
            )
        memory = self.output_norm(memory)
        if self.svd_rank is None:
            return memory

        # Preserve the common memory direction exactly and retain only the
        # leading centered slot-variation directions. Compute in float32
        # because CUDA SVD does not support bf16 and cast back afterwards.
        memory_dtype = memory.dtype
        memory_float = memory.float()
        memory_mean = memory_float.mean(dim=1, keepdim=True)
        centered = memory_float - memory_mean
        left, singular_values, right_t = torch.linalg.svd(
            centered,
            full_matrices=False,
        )
        rank = self.svd_rank
        centered_truncated = torch.matmul(
            left[..., :rank] * singular_values[..., :rank].unsqueeze(-2),
            right_t[..., :rank, :],
        )
        return (memory_mean + centered_truncated).to(dtype=memory_dtype)


class SerialDINOFeaturePredictor(nn.Module):
    """Predict future frozen-DINO patch features after the action expert has run."""

    def __init__(
        self,
        *,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        eps: float,
        num_heads: int,
        num_layers: int = 3,
        max_feature_tokens: int = 2048,
        max_action_tokens: int = 1024,
        action_projector_mlp_ratio: float = 2.0,
        add_feature_pos: bool = True,
        add_action_pos: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_feature_tokens = int(max_feature_tokens)
        self.max_action_tokens = int(max_action_tokens)
        self.add_feature_pos = bool(add_feature_pos)
        self.add_action_pos = bool(add_action_pos)
        self.feature_in = nn.Linear(self.feature_dim, self.hidden_dim)
        action_projector_hidden = max(
            self.hidden_dim,
            int(round(self.action_dim * float(action_projector_mlp_ratio))),
        )
        self.action_projector = nn.Sequential(
            nn.LayerNorm(self.action_dim, eps=float(eps)),
            nn.Linear(self.action_dim, action_projector_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_projector_hidden, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
        )
        self.blocks = nn.ModuleList(
            [
                SerialFeaturePredictionBlock(
                    hidden_dim=self.hidden_dim,
                    ffn_dim=int(ffn_dim),
                    eps=float(eps),
                    num_heads=self.num_heads,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.head = nn.Linear(self.hidden_dim, self.feature_dim)
        if self.add_feature_pos:
            self.feature_pos = nn.Parameter(torch.zeros(1, self.max_feature_tokens, self.hidden_dim))
            nn.init.trunc_normal_(self.feature_pos, std=0.02)
        else:
            self.register_parameter("feature_pos", None)
        if self.add_action_pos:
            self.action_pos = nn.Parameter(torch.zeros(1, self.max_action_tokens, self.hidden_dim))
            nn.init.trunc_normal_(self.action_pos, std=0.02)
        else:
            self.register_parameter("action_pos", None)

    def forward(
        self,
        *,
        start_feature_tokens: torch.Tensor,
        action_hidden: torch.Tensor,
        action_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if start_feature_tokens.ndim != 3:
            raise ValueError(f"`start_feature_tokens` must be [B,N,D], got {tuple(start_feature_tokens.shape)}")
        if action_hidden.ndim != 3:
            raise ValueError(f"`action_hidden` must be [B,S,D], got {tuple(action_hidden.shape)}")
        if start_feature_tokens.shape[2] != self.feature_dim:
            raise ValueError(
                f"`start_feature_tokens` last dim must be {self.feature_dim}, got {start_feature_tokens.shape[2]}"
            )
        if action_hidden.shape[2] != self.action_dim:
            raise ValueError(f"`action_hidden` last dim must be {self.action_dim}, got {action_hidden.shape[2]}")
        if action_hidden.shape[1] <= 0:
            raise ValueError("Serial DINO feature predictor received no action context tokens.")

        feature_len = int(start_feature_tokens.shape[1])
        action_len = int(action_hidden.shape[1])
        if feature_len > self.max_feature_tokens:
            raise ValueError(f"feature_len={feature_len} exceeds max_feature_tokens={self.max_feature_tokens}.")
        if action_positions is not None:
            if action_positions.ndim != 1 or action_positions.numel() != action_len:
                raise ValueError(
                    "`action_positions` must be [action_len], got "
                    f"{tuple(action_positions.shape)} for action_len={action_len}."
                )
            if action_positions.dtype != torch.long:
                raise ValueError(f"action_positions must use torch.long, got {action_positions.dtype}.")
        elif action_len > self.max_action_tokens:
            raise ValueError(f"action_len={action_len} exceeds max_action_tokens={self.max_action_tokens}.")

        x = self.feature_in(start_feature_tokens)
        if self.feature_pos is not None:
            x = x + self.feature_pos[:, :feature_len].to(device=x.device, dtype=x.dtype)

        action_context = self.action_projector(action_hidden)
        if self.action_pos is not None:
            if action_positions is None:
                pos = self.action_pos[:, :action_len]
            else:
                pos = self.action_pos[:, action_positions.to(device=self.action_pos.device)]
            action_context = action_context + pos.to(device=action_context.device, dtype=action_context.dtype)

        for block in self.blocks:
            x = block(x, action_context)
        return self.head(self.output_norm(x))


class SETFlareWAM(HistoryAwareWAM):
    """SETWAM variant with a serial frozen-feature prediction auxiliary loss.

    This class intentionally lives in a separate file from `aed_core.py` so
    transport-SET checkpoints can continue to load/eval against their original
    implementation. This variant keeps the action denoising path as history
    prefix + action tokens. During training, it first runs the regular
    video/action MoT, then uses final-layer action hidden states from a sampled
    action span to predict frozen DINO features at a future observation step.
    The feature predictor is training-only; action inference uses the same
    history-prefix action path and does not require future features.
    """

    def __init__(
        self,
        *args,
        set_config: Optional[dict[str, Any]] = None,
        loss_lambda_set: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.history_prefix_scale = float(self.history_config.get("prefix_scale", 1.0))
        if not math.isfinite(self.history_prefix_scale):
            raise ValueError(
                "`history.prefix_scale` must be finite, got "
                f"{self.history_prefix_scale}."
            )
        if set_config is None and float(loss_lambda_set) == 0.0:
            self.set_config = {}
            self.set_enabled = False
            self.set_enable_without_loss = False
            self.loss_lambda_set = 0.0
            self.set_num_event_queries = 0
            self.flare_use_history_prefix = False
            self.feature_enabled = False
            self.history_visual_memory_enabled = False
            self.history_visual_memory = None
        else:
            self.configure_flare_set(set_config=set_config, loss_lambda_set=loss_lambda_set)

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        set_config: Optional[dict[str, Any]] = None,
        loss_lambda_set: float = 0.0,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(*args, **kwargs)
        model.configure_flare_set(set_config=set_config, loss_lambda_set=loss_lambda_set)
        return model

    def build_inputs(self, sample, tiled: bool = False):
        text_memory = getattr(self, "history_visual_memory", None)
        text_block_index = (
            None
            if text_memory is None
            else text_memory.text_cross_attention_block_index
        )
        text_block_is_active = (
            text_block_index is not None
            and text_block_index not in text_memory.skip_block_indices
        )
        if not text_block_is_active:
            return super().build_inputs(sample, tiled=tiled)
        text_context = sample.get("context", None)
        text_context_mask = sample.get("context_mask", None)
        if text_context is None or text_context_mask is None:
            raise ValueError(
                "SET-FLARE text-conditioned memory requires training samples to "
                "provide both `context` and `context_mask`."
            )
        if text_context.ndim != 3 or text_context_mask.ndim != 2:
            raise ValueError(
                "Training text context must be [B,L,D] with mask [B,L], got "
                f"{tuple(text_context.shape)} and {tuple(text_context_mask.shape)}."
            )
        if text_context_mask.shape != text_context.shape[:2]:
            raise ValueError(
                "Training text mask must match the first two text dimensions, got "
                f"{tuple(text_context_mask.shape)} vs {tuple(text_context.shape)}."
            )
        effective_text_context_mask = text_context_mask.to(dtype=torch.bool) & (
            text_context.detach().abs().amax(dim=-1) > 0
        )
        if not bool(effective_text_context_mask.any(dim=1).all()):
            raise ValueError(
                "Every training sample must contain at least one valid text token."
            )
        inputs = super().build_inputs(sample, tiled=tiled)
        text_length = int(text_context.shape[1])
        if inputs["context"].shape[1] < text_length:
            raise ValueError(
                "Prepared model context is shorter than the original text context: "
                f"{inputs['context'].shape[1]} < {text_length}."
            )
        # `WAM.build_inputs` may append one projected proprio token. Preserve
        # an exact view of the original language tokens for the memory block.
        inputs["text_context"] = inputs["context"][:, :text_length]
        inputs["text_context_mask"] = effective_text_context_mask.to(
            device=inputs["context"].device,
            dtype=torch.bool,
            non_blocking=True,
        )
        return inputs

    def configure_flare_set(self, *, set_config: Optional[dict[str, Any]], loss_lambda_set: float) -> None:
        cfg = dict(set_config or {})
        self.set_config = cfg
        requested_enabled = bool(cfg.get("enabled", True))
        self.set_enable_without_loss = bool(cfg.get("enable_without_loss", False))
        self.loss_lambda_set = float(loss_lambda_set)
        # Keep the historical default (lambda_set <= 0 disables SET-FLARE), but
        # allow a loss-only ablation to retain the identical memory/action path.
        # In that ablation, the raw feature objective is still evaluated so all
        # predictor parameters stay in the distributed graph, while its weighted
        # contribution to the optimization objective remains exactly zero.
        self.set_enabled = requested_enabled and (
            self.loss_lambda_set > 0.0 or self.set_enable_without_loss
        )
        self.feature_enabled = self.set_enabled
        self.set_num_event_queries = 0
        self.history_visual_memory_enabled = False
        self.history_visual_memory = None
        self.history_visual_memory_allow_checkpoint_architecture_mismatch = False
        if not self.set_enabled:
            self.flare_use_history_prefix = False
            logger.info(
                "SET-FLARE feature-flow disabled: requested_enabled=%s "
                "enable_without_loss=%s lambda_set=%.4g",
                requested_enabled,
                self.set_enable_without_loss,
                self.loss_lambda_set,
            )
            return
        self.flare_use_history_prefix = bool(cfg.get("use_history_prefix", True))
        self.flare_detach_history_prefix = bool(cfg.get("detach_history_prefix", False))
        self.feature_attend_history_prefix = bool(cfg.get("feature_attend_history_prefix", False))
        self.feature_prediction_target = str(cfg.get("feature_prediction_target", "delta")).lower()
        if self.feature_prediction_target in {"diff", "difference", "change", "delta_state"}:
            self.feature_prediction_target = "delta"
        elif self.feature_prediction_target in {"future", "future_feature", "future_state", "end", "end_state"}:
            self.feature_prediction_target = "future"
        if self.feature_prediction_target not in {"delta", "future"}:
            raise ValueError(
                "`set.feature_prediction_target` must be 'delta' or 'future', got "
                f"{self.feature_prediction_target!r}."
            )
        self.feature_sampling_mode = str(cfg.get("feature_sampling_mode", "random")).lower()
        if self.feature_sampling_mode in {"sample", "random_span", "random_start_span"}:
            self.feature_sampling_mode = "random"
        elif self.feature_sampling_mode in {"fixed", "final", "full", "current_to_final"}:
            self.feature_sampling_mode = "fixed_final"
        if self.feature_sampling_mode not in {"random", "fixed_final"}:
            raise ValueError(
                "`set.feature_sampling_mode` must be 'random' or 'fixed_final', got "
                f"{self.feature_sampling_mode!r}."
            )
        self.feature_normalize_target = bool(cfg.get("normalize_target", True))
        if "feature_loss_type" in cfg:
            self.feature_loss_type = _normalize_feature_loss_type(cfg["feature_loss_type"])
        else:
            self.feature_loss_type = (
                "normalized_mse" if self.feature_normalize_target else "raw_mse"
            )
        self.feature_cosine_weight = float(cfg.get("cosine_weight", 0.0))
        if self.feature_loss_type in {
            "hwel_diagonal",
            "hwel_diagonal_batch",
        } and self.feature_cosine_weight != 0.0:
            raise ValueError(
                "Diagonal HWEL is a single endpoint objective; `set.cosine_weight` must be 0."
            )
        hwel_cfg = dict(cfg.get("hwel", {}))
        self.feature_hwel_ridge_relative = float(hwel_cfg.get("ridge_relative", 1.0e-6))
        self.feature_hwel_ridge_absolute = float(hwel_cfg.get("ridge_absolute", 1.0e-12))
        if self.feature_hwel_ridge_relative < 0.0 or self.feature_hwel_ridge_absolute <= 0.0:
            raise ValueError(
                "HWEL ridge values require ridge_relative >= 0 and ridge_absolute > 0, got "
                f"{self.feature_hwel_ridge_relative} and {self.feature_hwel_ridge_absolute}."
            )
        raw_hwel_profile_path = hwel_cfg.get("profile_path")
        self.feature_hwel_profile_path = (
            None
            if raw_hwel_profile_path is None or not str(raw_hwel_profile_path).strip()
            else str(raw_hwel_profile_path)
        )
        self.feature_hwel_profile_metadata: dict[str, Any] = {}
        self.feature_hwel_profile_sha256: Optional[str] = None
        self.register_buffer(
            "feature_hwel_inv_second_moment",
            torch.empty(0, 0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_hwel_normalizer_z",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_hwel_span_available",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.feature_min_action_span = int(cfg.get("min_action_span", 1))
        self.feature_max_action_span = int(cfg.get("max_action_span", cfg.get("max_feature_span", 32)))
        if self.feature_min_action_span <= 0:
            raise ValueError(f"`set.min_action_span` must be positive, got {self.feature_min_action_span}.")
        if self.feature_max_action_span < self.feature_min_action_span:
            raise ValueError(
                "`set.max_action_span` must be >= min_action_span, got "
                f"{self.feature_max_action_span} < {self.feature_min_action_span}."
            )

        target_cfg = dict(cfg.get("target", {}))
        self.feature_target_token_mode = str(target_cfg.get("token_mode", "patch")).lower()
        if self.feature_target_token_mode in {"patches", "patch_tokens"}:
            self.feature_target_token_mode = "patch"
        elif self.feature_target_token_mode in {"state", "global", "pool", "pooled"}:
            self.feature_target_token_mode = "pooled"
        if self.feature_target_token_mode not in {"patch", "pooled"}:
            raise ValueError(
                "`set.target.token_mode` must be 'patch' or 'pooled', got "
                f"{self.feature_target_token_mode!r}."
            )

        action_hidden_dim = int(self.action_expert.hidden_dim)
        hidden_dim = int(cfg.get("feature_hidden_dim", action_hidden_dim))
        feature_dim = int(cfg.get("feature_dim", hidden_dim))
        eps = float(cfg.get("eps", getattr(self.action_expert.blocks[0], "norm1").eps))
        ffn_dim = int(cfg.get("feature_ffn_dim", max(hidden_dim * 4, int(self.action_expert.ffn_dim))))
        action_projector_mlp_ratio = float(
            cfg.get("serial_action_projector_mlp_ratio", cfg.get("action_projector_mlp_ratio", 2.0))
        )
        self.serial_feature_predictor = SerialDINOFeaturePredictor(
            feature_dim=feature_dim,
            action_dim=action_hidden_dim,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            eps=eps,
            num_heads=int(cfg.get("serial_num_heads", 8)),
            num_layers=int(cfg.get("serial_num_layers", 3)),
            max_feature_tokens=int(cfg.get("serial_max_feature_tokens", 2048)),
            max_action_tokens=int(cfg.get("serial_max_action_tokens", 1024)),
            action_projector_mlp_ratio=action_projector_mlp_ratio,
            add_feature_pos=bool(cfg.get("serial_add_feature_pos", True)),
            add_action_pos=bool(cfg.get("serial_add_action_pos", True)),
        )

        target_type = str(target_cfg.get("type", "frozen_dino")).lower()
        dino_target_types = {"frozen_dino", "dino", "frozen_vision"}
        lingbot_target_types = {"frozen_lingbot", "lingbot", "lingbot_vision", "frozen_lingbot_vision"}
        if target_type not in dino_target_types | lingbot_target_types:
            raise ValueError(
                "SETFlareWAM requires a frozen DINOv3 or LingBot-Vision target, "
                f"got {target_type!r}."
            )
        self.set_target_type = target_type
        self.set_target_family = (
            "frozen_lingbot_vision" if target_type in lingbot_target_types else "frozen_dino"
        )
        target_cls = (
            SETFrozenLingBotVisionTarget
            if target_type in lingbot_target_types
            else SETFrozenVisionStateTarget
        )
        target_kwargs = {
            "hidden_dim": feature_dim,
            "model_id": str(
                target_cfg.get(
                    "model_id",
                    "robbyant/lingbot-vision-vit-base"
                    if target_type in lingbot_target_types
                    else "facebook/dinov3-vitb16-pretrain-lvd1689m",
                )
            ),
            "source": str(target_cfg.get("source", "huggingface")),
            "local_path": target_cfg.get("local_path", None),
            "cache_dir": target_cfg.get("cache_dir", None),
            "image_size": int(target_cfg.get("image_size", 224)),
            "input_range": str(target_cfg.get("input_range", "minus_one_one")),
            "pool": str(target_cfg.get("pool", "patch_mean")),
            "eps": eps,
            "projection_seed": int(target_cfg.get("projection_seed", 0)),
            "torch_dtype": self.torch_dtype,
        }
        self.set_target_model_id = str(target_kwargs["model_id"])
        self.set_target_feature_spec = frozen_target_feature_spec(
            {
                "feature_dim": feature_dim,
                "eps": eps,
                "target": target_cfg,
            },
            verify_local_model=True,
        )
        if target_type in lingbot_target_types:
            target_kwargs["variant"] = str(target_cfg.get("variant", "base"))
        self.set_target_encoder = target_cls(**target_kwargs)
        self.serial_feature_predictor.to(device=self.device, dtype=self.torch_dtype)
        self.set_target_encoder.to(device=self.device, dtype=self.torch_dtype)
        self.set_target_encoder.eval().requires_grad_(False)
        if self.feature_loss_type == "hwel_diagonal":
            if self.feature_hwel_profile_path is None:
                logger.warning(
                    "Diagonal HWEL profile is not configured. Model construction is allowed for "
                    "inference, but training_loss will fail until a profile is loaded."
                )
            elif Path(self.feature_hwel_profile_path).expanduser().is_file():
                self._install_hwel_profile_from_path(self.feature_hwel_profile_path)
            else:
                logger.warning(
                    "Diagonal HWEL profile path is unavailable at model construction: %s. "
                    "Inference still works; training_loss will raise an error.",
                    self.feature_hwel_profile_path,
                )

        history_source = str(self.history_config.get("visual_token_source", "vae")).lower()
        self.history_visual_memory_enabled = history_source in (
            _DINO_MEMORY_SOURCES | _VAE_MEMORY_SOURCES
        )
        self.history_visual_memory = None
        if self.history_visual_memory_enabled:
            memory_cfg = dict(self.history_config.get("visual_memory", {}))
            self.history_visual_memory_allow_checkpoint_architecture_mismatch = bool(
                memory_cfg.get("allow_checkpoint_architecture_mismatch", False)
            )
            self.history_visual_memory_include_current = bool(memory_cfg.get("include_current", True))
            memory_input_dim = (
                action_hidden_dim if history_source in _VAE_MEMORY_SOURCES else feature_dim
            )
            self.history_visual_memory = HistoryVisualMemoryExtractor(
                input_dim=memory_input_dim,
                hidden_dim=action_hidden_dim,
                num_queries=int(memory_cfg.get("num_queries", 128)),
                num_layers=int(memory_cfg.get("num_layers", 2)),
                num_heads=int(memory_cfg.get("num_heads", 8)),
                ffn_dim=int(memory_cfg.get("ffn_dim", action_hidden_dim * 4)),
                max_frames=int(memory_cfg.get("max_frames", max(33, self.history_video_frames))),
                eps=float(memory_cfg.get("eps", eps)),
                skip_block_indices=[
                    int(index)
                    for index in memory_cfg.get("skip_block_indices", [])
                ],
                svd_rank=(
                    None
                    if memory_cfg.get("svd_rank") is None
                    else int(memory_cfg.get("svd_rank"))
                ),
                text_input_dim=(
                    None
                    if memory_cfg.get("text_cross_attention_block_index") is None
                    else int(self.text_dim)
                ),
                text_cross_attention_block_index=(
                    None
                    if memory_cfg.get("text_cross_attention_block_index") is None
                    else int(memory_cfg.get("text_cross_attention_block_index"))
                ),
            )
            self.history_visual_memory.to(device=self.device, dtype=self.torch_dtype)
        self._install_feature_mot()
        logger.info(
            "SET-FLARE serial-feature WAM config: enabled=%s "
            "enable_without_loss=%s lambda_set=%.4g "
            "feature_dim=%d hidden_dim=%d serial_layers=%d serial_heads=%d "
            "action_projector_mlp_ratio=%.3g use_history_prefix=%s detach_history_prefix=%s span=[%d,%d] "
            "target_token_mode=%s feature_prediction_target=%s feature_sampling_mode=%s "
            "loss_type=%s normalize_target=%s cosine_weight=%.4g hwel_profile=%s hwel_sha256=%s",
            self.set_enabled,
            self.set_enable_without_loss,
            self.loss_lambda_set,
            feature_dim,
            hidden_dim,
            int(cfg.get("serial_num_layers", 3)),
            int(cfg.get("serial_num_heads", 8)),
            action_projector_mlp_ratio,
            self.flare_use_history_prefix,
            self.flare_detach_history_prefix,
            self.feature_min_action_span,
            self.feature_max_action_span,
            self.feature_target_token_mode,
            self.feature_prediction_target,
            self.feature_sampling_mode,
            self.feature_loss_type,
            self.feature_normalize_target,
            self.feature_cosine_weight,
            self.feature_hwel_profile_path,
            self.feature_hwel_profile_sha256,
        )
        if self.history_visual_memory_enabled:
            logger.info(
                "SET-FLARE adaptive %s history memory: queries=%d layers=%d max_frames=%d "
                "include_current=%s skip_blocks=%s svd_rank=%s text_cross_block=%s "
                "text_input_dim=%s text_projector=%s",
                "VAE" if history_source in _VAE_MEMORY_SOURCES else "DINO",
                self.history_visual_memory.num_queries,
                self.history_visual_memory.num_layers,
                self.history_visual_memory.max_frames,
                self.history_visual_memory_include_current,
                sorted(self.history_visual_memory.skip_block_indices),
                self.history_visual_memory.svd_rank,
                self.history_visual_memory.text_cross_attention_block_index,
                self.history_visual_memory.text_input_dim,
                (
                    None
                    if self.history_visual_memory.text_projector is None
                    else type(self.history_visual_memory.text_projector).__name__
                ),
            )
        logger.info(
            "SET-FLARE history prefix scale: configured=%.4g effective=%.4g",
            self.history_prefix_scale,
            self._resolve_history_prefix_scale(),
        )

    def _set_hwel_profile_tensors(
        self,
        *,
        inv_second_moment: torch.Tensor,
        normalizer_z: torch.Tensor,
        span_available: torch.Tensor,
        metadata: Optional[dict[str, Any]] = None,
        sha256: Optional[str] = None,
    ) -> None:
        if inv_second_moment.ndim != 2:
            raise ValueError(
                "HWEL inverse-second-moment table must be [span, feature_dim], got "
                f"{tuple(inv_second_moment.shape)}."
            )
        if normalizer_z.shape != (inv_second_moment.shape[0],):
            raise ValueError(
                "HWEL normalizer table must match the span table, got "
                f"{tuple(normalizer_z.shape)} vs {tuple(inv_second_moment.shape)}."
            )
        if span_available.shape != normalizer_z.shape:
            raise ValueError(
                "HWEL span-availability table must match normalizers, got "
                f"{tuple(span_available.shape)} vs {tuple(normalizer_z.shape)}."
            )
        if inv_second_moment.shape[1] != int(self.serial_feature_predictor.feature_dim):
            raise ValueError(
                "HWEL profile feature dimension does not match the predictor: "
                f"{inv_second_moment.shape[1]} vs {self.serial_feature_predictor.feature_dim}."
            )
        available_inv = inv_second_moment[span_available.bool()]
        available_z = normalizer_z[span_available.bool()]
        if available_inv.numel() == 0:
            raise ValueError("HWEL profile contains no available spans.")
        if not bool(torch.isfinite(available_inv).all()) or bool((available_inv <= 0).any()):
            raise ValueError("HWEL inverse-second-moment values must be finite and positive.")
        if not bool(torch.isfinite(available_z).all()) or bool((available_z <= 0).any()):
            raise ValueError("HWEL normalizers must be finite and positive.")
        device = torch.device(self.device)
        self.feature_hwel_inv_second_moment = inv_second_moment.detach().to(
            device=device,
            dtype=torch.float32,
        )
        self.feature_hwel_normalizer_z = normalizer_z.detach().to(
            device=device,
            dtype=torch.float32,
        )
        self.feature_hwel_span_available = span_available.detach().to(
            device=device,
            dtype=torch.bool,
        )
        self.feature_hwel_profile_metadata = dict(metadata or {})
        self.feature_hwel_profile_sha256 = None if sha256 is None else str(sha256)

    def _install_hwel_profile_from_path(self, path: str | os.PathLike[str]) -> None:
        profile = _load_diagonal_hwel_profile(
            path,
            expected_feature_dim=int(self.serial_feature_predictor.feature_dim),
            expected_token_mode=self.feature_target_token_mode,
            expected_target_type=self.set_target_family,
            expected_model_id=self.set_target_model_id,
            expected_feature_spec=self.set_target_feature_spec,
        )
        inv_second_moment, normalizer_z, span_available, metadata, sha256 = profile
        self._set_hwel_profile_tensors(
            inv_second_moment=inv_second_moment,
            normalizer_z=normalizer_z,
            span_available=span_available,
            metadata=metadata,
            sha256=sha256,
        )
        logger.info(
            "Loaded diagonal HWEL profile: path=%s sha256=%s spans=%s",
            Path(path).expanduser().resolve(),
            sha256,
            torch.nonzero(span_available, as_tuple=False).flatten().tolist(),
        )

    def _hwel_span_geometry(
        self,
        *,
        span: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_hwel_span_available.numel() == 0:
            raise RuntimeError(
                "Diagonal HWEL training requires a loaded profile. Configure "
                "`model.aed.hwel.profile_path` with a compatible profile."
            )
        span = int(span)
        if (
            span <= 0
            or span >= int(self.feature_hwel_span_available.shape[0])
            or not bool(self.feature_hwel_span_available[span].item())
        ):
            available = torch.nonzero(
                self.feature_hwel_span_available,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                f"HWEL profile has no statistics for sampled span={span}; "
                f"available spans={available}."
            )
        return (
            self.feature_hwel_inv_second_moment[span].to(device=device, dtype=torch.float32),
            self.feature_hwel_normalizer_z[span].to(device=device, dtype=torch.float32),
        )

    def _install_feature_mot(self) -> None:
        previous_mot = getattr(self, "mot", None)
        checkpoint_mixed_attn = bool(getattr(previous_mot, "mot_checkpoint_mixed_attn", True))
        self.mot = MoT(
            mixtures={
                "video": self.video_expert,
                "action": self.action_expert,
            },
            mot_checkpoint_mixed_attn=checkpoint_mixed_attn,
        )
        self.dit = self.mot

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "serial_feature_predictor"):
            self.serial_feature_predictor.to(*args, **kwargs)
        if getattr(self, "set_target_encoder", None) is not None:
            self.set_target_encoder.to(*args, **kwargs)
            self.set_target_encoder.eval().requires_grad_(False)
        if getattr(self, "history_visual_memory", None) is not None:
            self.history_visual_memory.to(*args, **kwargs)
        return self

    def extra_trainable_modules(self):
        if not hasattr(self, "serial_feature_predictor"):
            return list(super().extra_trainable_modules())
        modules = list(super().extra_trainable_modules()) if self.flare_use_history_prefix else []
        modules.append(self.serial_feature_predictor)
        if getattr(self, "history_visual_memory", None) is not None:
            modules.append(self.history_visual_memory)
        return modules

    def extra_trainable_parameters(self):
        if not hasattr(self, "serial_feature_predictor"):
            return list(super().extra_trainable_parameters())
        params = list(super().extra_trainable_parameters()) if self.flare_use_history_prefix else []
        params.extend(self.serial_feature_predictor.parameters())
        if getattr(self, "history_visual_memory", None) is not None:
            params.extend(self.history_visual_memory.parameters())
        return params

    def _frozen_video_state(self, video: torch.Tensor, *, step: int) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"SET-FLARE frozen target expects video [B,3,T,H,W], got {tuple(video.shape)}")
        if step < 0:
            step = int(video.shape[2]) + step
        if not (0 <= step < int(video.shape[2])):
            raise IndexError(f"SET-FLARE frozen target frame step {step} out of range for video T={video.shape[2]}.")
        frame = video[:, :, step].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        state = self.set_target_encoder(frame)
        return state.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _frozen_image_patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"SET-FLARE patch target expects frames [B,3,H,W], got {tuple(frames.shape)}")
        return self.set_target_encoder.patch_tokens(
            frames.to(device=self.device, dtype=self.torch_dtype)
        )

    @torch.no_grad()
    def _frozen_image_feature_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        if self.feature_target_token_mode == "patch":
            return self._frozen_image_patch_tokens(frames)
        state = self.set_target_encoder(frames.to(device=self.device, dtype=self.torch_dtype))
        return state.unsqueeze(1).detach()

    def _history_video_tokens(
        self,
        *,
        history_video: torch.Tensor,
        tiled: bool,
        precomputed_dino_tokens: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        text_context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source = str(self.history_config.get("visual_token_source", "vae")).lower()
        if source in {"vae", "vae_first", "latent", "latent_first"}:
            return super()._history_video_tokens(history_video=history_video, tiled=tiled)
        if source in _VAE_MEMORY_SOURCES:
            if getattr(self, "history_visual_memory", None) is None:
                raise RuntimeError(
                    "history.visual_token_source=vae_memory requires the adaptive history visual memory module."
                )
            frames = history_video
            if not self.history_visual_memory_include_current:
                if frames.shape[2] <= 1:
                    raise ValueError("Cannot exclude the current frame from a one-frame VAE history window.")
                frames = frames[:, :, :-1]
            history_latents = self._encode_video_latents(frames, tiled=tiled)
            history_tokens = self.history_visual_tokenizer(history_latents)
            patch_t, patch_h, patch_w = self.history_visual_tokenizer.patch_size
            latent_t, latent_h, latent_w = history_latents.shape[2:]
            token_frames = latent_t // patch_t
            patches_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
            expected_tokens = token_frames * patches_per_frame
            if history_tokens.shape[1] != expected_tokens:
                raise ValueError(
                    "VAE history token layout mismatch: got "
                    f"{history_tokens.shape[1]} tokens, expected {token_frames}x{patches_per_frame}."
                )
            frame_patch_tokens = history_tokens.reshape(
                history_tokens.shape[0],
                token_frames,
                patches_per_frame,
                history_tokens.shape[-1],
            )
            return self.history_visual_memory(
                frame_patch_tokens,
                text_context=text_context,
                text_context_mask=text_context_mask,
            )
        if source not in {"dino", "dinov3", "frozen_dino"} | _DINO_MEMORY_SOURCES:
            raise ValueError(
                "`history.visual_token_source` must be 'vae', 'vae_memory', 'dino', or "
                "'dino_memory' for SET-FLARE, "
                f"got {source!r}."
            )
        if getattr(self, "set_target_encoder", None) is None:
            raise RuntimeError("history.visual_token_source=dino requires SET-FLARE frozen DINO target encoder.")
        if history_video.ndim != 5 or history_video.shape[1] != 3:
            raise ValueError(f"DINO history tokens expect history_video [B,3,T,H,W], got {tuple(history_video.shape)}")
        if precomputed_dino_tokens is not None:
            if precomputed_dino_tokens.ndim != 4:
                raise ValueError(
                    "Cached history DINO tokens must be [B,T,N,D], got "
                    f"{tuple(precomputed_dino_tokens.shape)}."
                )
            if precomputed_dino_tokens.shape[:2] != (history_video.shape[0], history_video.shape[2]):
                raise ValueError(
                    "Cached history DINO batch/time dimensions do not match history video: "
                    f"{tuple(precomputed_dino_tokens.shape)} vs {tuple(history_video.shape)}."
                )
        if source in _DINO_MEMORY_SOURCES:
            if getattr(self, "history_visual_memory", None) is None:
                raise RuntimeError(
                    "history.visual_token_source=dino_memory requires the adaptive history visual memory module."
                )
            if precomputed_dino_tokens is not None:
                patch_tokens = precomputed_dino_tokens
                if not self.history_visual_memory_include_current:
                    if patch_tokens.shape[1] <= 1:
                        raise ValueError("Cannot exclude the current frame from a one-frame DINO history window.")
                    patch_tokens = patch_tokens[:, :-1]
                return self.history_visual_memory(
                    patch_tokens,
                    text_context=text_context,
                    text_context_mask=text_context_mask,
                )

            frames = history_video
            if not self.history_visual_memory_include_current:
                if frames.shape[2] <= 1:
                    raise ValueError("Cannot exclude the current frame from a one-frame DINO history window.")
                frames = frames[:, :, :-1]
            batch, channels, num_frames, height, width = frames.shape
            flat_frames = frames.permute(0, 2, 1, 3, 4).reshape(
                batch * num_frames,
                channels,
                height,
                width,
            )
            patch_tokens = self._frozen_image_patch_tokens(flat_frames)
            patch_tokens = patch_tokens.reshape(
                batch,
                num_frames,
                patch_tokens.shape[1],
                patch_tokens.shape[2],
            )
            return self.history_visual_memory(
                patch_tokens,
                text_context=text_context,
                text_context_mask=text_context_mask,
            )

        frame_selector = str(self.history_config.get("dino_history_frame", "current")).lower()
        if frame_selector in {"current", "last"}:
            frame_index = -1
        elif frame_selector in {"first", "oldest", "start"}:
            frame_index = 0
        else:
            frame_index = int(frame_selector)
        if precomputed_dino_tokens is not None:
            return precomputed_dino_tokens[:, frame_index].to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        frame = history_video[:, :, frame_index].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        return self._frozen_image_feature_tokens(frame).to(device=self.device, dtype=self.torch_dtype).detach()

    def _empty_history_prefix(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((batch_size, 0, int(self.action_expert.hidden_dim)), device=device, dtype=dtype)

    def _resolve_history_prefix_scale(self) -> float:
        scale = float(os.environ.get("WAM_HISTORY_PREFIX_SCALE", self.history_prefix_scale))
        if not math.isfinite(scale):
            raise ValueError(f"SET-FLARE history prefix scale must be finite, got {scale}.")
        return scale

    def _apply_history_prefix_scale(self, history_prefix: torch.Tensor) -> torch.Tensor:
        scale = self._resolve_history_prefix_scale()
        if scale == 1.0:
            return history_prefix
        return history_prefix * scale

    @staticmethod
    def _validate_dino_history_alignment(
        *,
        action_horizon: int,
        group_size: int,
        num_visual_frames: int,
    ) -> int:
        if group_size <= 0 or action_horizon % group_size != 0:
            raise ValueError(
                "DINO history memory requires action_horizon divisible by K, got "
                f"{action_horizon} actions and K={group_size}."
            )
        action_groups = action_horizon // group_size
        visual_intervals = num_visual_frames - 1
        if action_groups != visual_intervals:
            raise ValueError(
                "DINO history temporal mismatch: grouped actions must match visual intervals, got "
                f"{action_groups} action groups but {visual_intervals} intervals from "
                f"{num_visual_frames} frames."
            )
        return action_groups

    def _maybe_history_prefix(
        self,
        *,
        history_video: torch.Tensor,
        history_action: torch.Tensor,
        history_state: Optional[torch.Tensor],
        tiled: bool,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        history_dino_tokens: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        text_context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.flare_use_history_prefix:
            return self._empty_history_prefix(batch_size=batch_size, device=device, dtype=dtype)
        history_video, history_action, history_start_state, _ = self._prepare_history_context(
            history_video=history_video,
            history_action=history_action,
            history_state=history_state,
        )
        if (
            getattr(self, "history_visual_memory", None) is not None
            and self.history_adapter.action_one_token_per_group
        ):
            group_size = int(self.history_adapter.action_sum_group_size)
            action_horizon = int(history_action.shape[1])
            self._validate_dino_history_alignment(
                action_horizon=action_horizon,
                group_size=group_size,
                num_visual_frames=int(history_video.shape[2]),
            )
        history_tokens = self._history_video_tokens(
            history_video=history_video,
            tiled=tiled,
            precomputed_dino_tokens=history_dino_tokens,
            text_context=text_context,
            text_context_mask=text_context_mask,
        )
        history_prefix, _ = self.history_adapter(
            history_action,
            history_tokens,
            history_start_state=history_start_state,
        )
        history_prefix = history_prefix.to(device=device, dtype=dtype)
        if self.flare_detach_history_prefix:
            history_prefix = history_prefix.detach()
        return self._apply_history_prefix_scale(history_prefix)

    def _concat_feature_action_pre(self, action_pre: dict[str, Any], history_prefix: torch.Tensor) -> dict[str, Any]:
        prefix_len = int(history_prefix.shape[1])
        action_tokens = action_pre["tokens"]
        action_len = int(action_tokens.shape[1])
        extra_len = action_len if self.extra_prediction_action_tokens else 0
        total_len = prefix_len + action_len + extra_len
        if total_len > self.action_expert.freqs.shape[0]:
            raise ValueError(f"SET-FLARE action length {total_len} exceeds RoPE cache {self.action_expert.freqs.shape[0]}.")

        action_t_mod = action_pre["t_mod"]
        if action_t_mod.ndim == 3:
            action_t_mod = action_t_mod.unsqueeze(1).expand(-1, action_len, -1, -1).contiguous()
        elif action_t_mod.ndim != 4:
            raise ValueError(f"Unsupported action t_mod shape: {tuple(action_t_mod.shape)}")

        context_mask = action_pre["context_mask"]
        if context_mask.ndim != 3:
            raise ValueError(f"`action_pre.context_mask` must be [B,S,L], got {tuple(context_mask.shape)}")

        prefix_t_mod = self._build_zero_action_t_mod(
            batch_size=action_tokens.shape[0],
            seq_len=prefix_len,
            dtype=action_tokens.dtype,
            device=action_tokens.device,
        )
        prefix_context_mask = context_mask[:, :1, :].expand(-1, prefix_len, -1)

        token_chunks = [history_prefix, action_tokens]
        t_mod_chunks = [prefix_t_mod, action_t_mod]
        context_mask_chunks = [prefix_context_mask, context_mask]
        if self.extra_prediction_action_tokens:
            token_chunks.append(action_tokens)
            t_mod_chunks.append(action_t_mod)
            context_mask_chunks.append(context_mask)

        current_start = prefix_len
        current_end = current_start + action_len
        extra_start = current_end
        extra_end = extra_start + extra_len
        prediction_start = extra_start if self.prediction_action_group == "extra" else current_start
        prediction_end = extra_end if self.prediction_action_group == "extra" else current_end

        merged = dict(action_pre)
        merged["tokens"] = torch.cat(token_chunks, dim=1)
        merged["freqs"] = self.action_expert.freqs[:total_len].view(total_len, 1, -1).to(action_tokens.device)
        merged["t_mod"] = torch.cat(t_mod_chunks, dim=1)
        merged["context_mask"] = torch.cat(context_mask_chunks, dim=1)
        merged["meta"] = dict(action_pre.get("meta", {}))
        merged["meta"].update(
            {
                "history_prefix_len": prefix_len,
                "current_action_len": action_len,
                "extra_action_len": extra_len,
                "current_action_start": current_start,
                "current_action_end": current_end,
                "extra_action_start": extra_start,
                "extra_action_end": extra_end,
                "prediction_action_group": self.prediction_action_group,
                "prediction_action_start": prediction_start,
                "prediction_action_end": prediction_end,
            }
        )
        return merged

    @staticmethod
    def _video_span_to_action_offsets(
        *,
        start_offset: int,
        span: int,
        current_action_len: int,
        current_frame_transitions: int,
    ) -> tuple[int, int, int]:
        if current_frame_transitions <= 0:
            raise ValueError(
                "`current_frame_transitions` must be positive, "
                f"got {current_frame_transitions}."
            )
        if current_action_len <= 0:
            raise ValueError(f"`current_action_len` must be positive, got {current_action_len}.")
        if current_action_len % current_frame_transitions != 0:
            raise ValueError(
                "Current action horizon must be divisible by the number of video transitions for "
                "temporally aligned feature prediction, got "
                f"actions={current_action_len}, video_transitions={current_frame_transitions}."
            )
        if start_offset < 0 or span <= 0 or start_offset + span > current_frame_transitions:
            raise ValueError(
                "Invalid sampled video span: "
                f"start_offset={start_offset}, span={span}, "
                f"video_transitions={current_frame_transitions}."
            )

        actions_per_video_interval = current_action_len // current_frame_transitions
        action_start_offset = start_offset * actions_per_video_interval
        action_end_offset = (start_offset + span) * actions_per_video_interval
        return action_start_offset, action_end_offset, actions_per_video_interval

    def _sample_feature_prediction_batch(
        self,
        *,
        video: torch.Tensor,
        history_frame_count: int,
        action_meta: dict[str, Any],
        current_action_len: int,
        dtype: torch.dtype,
        device: torch.device,
        video_feature_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if video.ndim != 5:
            raise ValueError(f"Serial feature prediction expects video [B,3,T,H,W], got {tuple(video.shape)}")
        if history_frame_count < 0 or video.shape[2] < 2:
            raise ValueError(
                "Serial feature prediction needs at least one history/current frame and one future frame; "
                f"history_frame_count={history_frame_count} video_T={video.shape[2]}"
            )

        num_frames = int(history_frame_count) + int(video.shape[2])
        current_frame_transitions = int(video.shape[2]) - 1
        if current_frame_transitions <= 0:
            raise ValueError(
                "Feature prediction found no current/future video transitions: "
                f"frames={num_frames}, history_frames={history_frame_count}."
            )
        if current_action_len % current_frame_transitions != 0:
            raise ValueError(
                "Current action horizon must be divisible by current/future video transitions, got "
                f"actions={current_action_len}, video_transitions={current_frame_transitions}."
            )
        max_possible_span = min(self.feature_max_action_span, current_frame_transitions)
        if max_possible_span < self.feature_min_action_span:
            raise ValueError(
                "No valid feature-prediction span: "
                f"max_span={max_possible_span}, min_span={self.feature_min_action_span}, "
                f"current_action_len={current_action_len}, current_frame_transitions={current_frame_transitions}, "
                f"frames={num_frames}, history_frames={history_frame_count}"
            )
        sampling_mode = getattr(self, "feature_sampling_mode", "random")
        if sampling_mode == "fixed_final":
            start_offset = 0
            span = current_frame_transitions
        else:
            max_start_offset = current_frame_transitions - self.feature_min_action_span
            start_offset = random.randint(0, max_start_offset)
            max_span_from_start = min(
                self.feature_max_action_span,
                current_frame_transitions - start_offset,
            )
            if max_span_from_start < self.feature_min_action_span:
                raise ValueError(
                    "Sampled feature-prediction start has no valid future span: "
                    f"start_offset={start_offset}, max_span_from_start={max_span_from_start}, "
                    f"min_span={self.feature_min_action_span}."
                )
            span = random.randint(self.feature_min_action_span, max_span_from_start)
        start = history_frame_count + start_offset
        end = start + span

        if video_feature_tokens is None:
            endpoint_frames = torch.cat(
                [video[:, :, start_offset], video[:, :, start_offset + span]],
                dim=0,
            )
            endpoint_states = self._frozen_image_feature_tokens(endpoint_frames).to(
                device=device,
                dtype=dtype,
            ).detach()
            start_state, end_state = endpoint_states.chunk(2, dim=0)
        else:
            if video_feature_tokens.ndim != 4 or video_feature_tokens.shape[:2] != (
                video.shape[0],
                video.shape[2],
            ):
                raise ValueError(
                    "Cached video DINO tokens must be [B,T,N,D] aligned with video, got "
                    f"{tuple(video_feature_tokens.shape)} vs {tuple(video.shape)}."
                )
            start_state = video_feature_tokens[:, start_offset].to(device=device, dtype=dtype).detach()
            end_state = video_feature_tokens[:, start_offset + span].to(device=device, dtype=dtype).detach()
        if start_state.ndim != 3 or end_state.ndim != 3:
            raise ValueError(
                "Feature-flow target tokens must be [B,N,D], got "
                f"{tuple(start_state.shape)} and {tuple(end_state.shape)}."
            )
        if start_state.shape != end_state.shape:
            raise ValueError(
                "Start/end feature token shapes must match for feature flow, got "
                f"{tuple(start_state.shape)} and {tuple(end_state.shape)}."
            )

        action_seq_len = int(action_meta["extra_action_end"])
        selected_action_mask = torch.zeros(action_seq_len, device=device, dtype=torch.bool)
        prediction_start = int(action_meta["prediction_action_start"])
        prediction_end = int(action_meta["prediction_action_end"])
        prediction_len = prediction_end - prediction_start
        if prediction_len < current_action_len:
            raise ValueError(
                "SET-FLARE prediction action group is shorter than current action length: "
                f"prediction_len={prediction_len}, current_action_len={current_action_len}."
            )
        action_start_offset, action_end_offset, actions_per_video_interval = self._video_span_to_action_offsets(
            start_offset=start_offset,
            span=span,
            current_action_len=current_action_len,
            current_frame_transitions=current_frame_transitions,
        )
        action_start = prediction_start + action_start_offset
        action_end = prediction_start + action_end_offset
        if action_end > prediction_end:
            raise ValueError(
                "Sampled feature span exceeds prediction action group: "
                f"action_start={action_start}, action_end={action_end}, prediction_end={prediction_end}."
            )
        selected_action_mask[action_start:action_end] = True
        if action_end <= action_start:
            raise ValueError("Feature-flow sampling selected no prediction action tokens.")

        target = end_state - start_state if self.feature_prediction_target == "delta" else end_state

        return {
            "start_state": start_state,
            "target": target.detach(),
            "target_state": end_state,
            "target_delta": (end_state - start_state).detach(),
            "prediction_target_mode": self.feature_prediction_target,
            "selected_action_mask": selected_action_mask,
            "span": span,
            "start": start,
            "end": end,
            "action_start": action_start,
            "action_end": action_end,
            "actions_per_video_interval": actions_per_video_interval,
        }

    def _select_serial_action_context(
        self,
        *,
        action_tokens: torch.Tensor,
        selected_action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,S,D], got {tuple(action_tokens.shape)}.")
        if selected_action_mask.ndim != 1 or selected_action_mask.shape[0] != action_tokens.shape[1]:
            raise ValueError(
                "`selected_action_mask` must be [S] matching action token length, got "
                f"{tuple(selected_action_mask.shape)} vs action_tokens={tuple(action_tokens.shape)}."
            )
        if action_tokens.shape[1] > self.serial_feature_predictor.max_action_tokens:
            raise ValueError(
                f"action token length {action_tokens.shape[1]} exceeds serial_max_action_tokens="
                f"{self.serial_feature_predictor.max_action_tokens}."
            )
        selected_indices = torch.nonzero(selected_action_mask.to(device=action_tokens.device), as_tuple=False).flatten()
        if selected_indices.numel() <= 0:
            raise ValueError("Serial feature predictor selected no action tokens.")
        return action_tokens[:, selected_indices], selected_indices

    def _build_feature_attention_mask(
        self,
        *,
        video_seq_len: int,
        action_seq_len: int,
        feature_seq_len: int,
        history_prefix_len: int,
        current_action_len: int,
        extra_action_len: int,
        selected_action_mask: torch.Tensor,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        base_mask = self._build_history_attention_mask(
            video_seq_len=video_seq_len,
            history_prefix_len=history_prefix_len,
            current_action_len=current_action_len,
            extra_action_len=extra_action_len,
            semantic_query_len=0,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        expected_base = video_seq_len + action_seq_len
        if base_mask.shape != (expected_base, expected_base):
            raise ValueError(
                f"Base action/video mask shape mismatch: {tuple(base_mask.shape)} vs {(expected_base, expected_base)}"
            )
        if selected_action_mask.ndim != 1 or selected_action_mask.shape[0] != action_seq_len:
            raise ValueError(
                "`selected_action_mask` must be [action_seq_len], got "
                f"{tuple(selected_action_mask.shape)} for action_seq_len={action_seq_len}"
            )
        total = expected_base + feature_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:expected_base, :expected_base] = base_mask
        feature_start = expected_base
        feature_end = feature_start + feature_seq_len
        mask[feature_start:feature_end, feature_start:feature_end] = True
        selected_cols = torch.nonzero(selected_action_mask.to(device=device), as_tuple=False).flatten()
        if selected_cols.numel() == 0:
            raise ValueError("Feature-flow attention selected no action tokens.")
        selected_cols = selected_cols + video_seq_len
        mask[feature_start:feature_end, selected_cols] = True
        return mask

    def _compute_feature_prediction_loss(
        self,
        *,
        pred_feature_tokens: torch.Tensor,
        feature_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.set_enabled:
            return pred_feature_tokens.new_zeros(()), {}
        target = feature_batch["target"]
        if target.ndim != 3:
            raise ValueError(f"Feature-prediction target must be [B,N,D], got {tuple(target.shape)}.")
        pred = pred_feature_tokens
        if pred.shape != target.shape:
            raise ValueError(
                "Feature-prediction prediction/target shape mismatch: "
                f"pred={tuple(pred.shape)} target={tuple(target.shape)}."
            )
        pred_float = pred.float()
        target_float = target.float()
        pred_normalized = F.normalize(pred_float, dim=-1)
        target_normalized = F.normalize(target_float, dim=-1)
        normalized_mse = F.mse_loss(pred_normalized, target_normalized)
        raw_target_mse = F.mse_loss(pred_float, target_float)
        loss_cos = (
            1.0 - (pred_normalized * target_normalized).sum(dim=-1)
        ).mean()

        prediction_target_mode = feature_batch.get(
            "prediction_target_mode",
            self.feature_prediction_target,
        )
        start_state = feature_batch["start_state"].detach().float()
        target_future = feature_batch["target_state"].detach().float()
        target_delta = feature_batch["target_delta"].detach().float()
        if prediction_target_mode == "delta":
            pred_delta = pred_float
            pred_future_for_grad = start_state + pred_float
        elif prediction_target_mode == "future":
            pred_future_for_grad = pred_float
            pred_delta = pred_float - start_state
        else:
            raise ValueError(f"Unsupported feature prediction target mode: {prediction_target_mode!r}.")
        endpoint_error = pred_future_for_grad - target_future
        delta_error = pred_delta - target_delta

        hwel_z = None
        hwel_nochange = None
        hwel_gain = None
        hwel_degenerate_fallback = None
        if self.feature_loss_type == "normalized_mse":
            loss_mse = normalized_mse
            loss = loss_mse + self.feature_cosine_weight * loss_cos
        elif self.feature_loss_type == "raw_mse":
            loss_mse = raw_target_mse
            loss = loss_mse + self.feature_cosine_weight * loss_cos
        elif self.feature_loss_type == "hwel_diagonal":
            inv_second_moment, hwel_z = self._hwel_span_geometry(
                span=int(feature_batch["span"]),
                device=delta_error.device,
            )
            loss = (
                delta_error.square() * inv_second_moment.view(1, 1, -1)
            ).sum(dim=-1).div(hwel_z).mean()
            hwel_nochange = (
                target_delta.square() * inv_second_moment.view(1, 1, -1)
            ).sum(dim=-1).div(hwel_z).mean()
            hwel_gain = 1.0 - loss / hwel_nochange.clamp_min(1.0e-12)
            loss_mse = raw_target_mse
        elif self.feature_loss_type == "hwel_diagonal_batch":
            # Estimate target-only geometry from the current batch. Patch tokens
            # provide B*N observations per channel, so training can start from the
            # first batch without a feature cache or an offline profile. The same
            # detached target moments define Z, preserving no-change calibration.
            batch_second_moment = target_delta.square().mean(dim=(0, 1))
            batch_mean_second_moment = batch_second_moment.mean()
            batch_ridge = torch.maximum(
                batch_mean_second_moment * float(self.feature_hwel_ridge_relative),
                batch_mean_second_moment.new_tensor(self.feature_hwel_ridge_absolute),
            )
            inv_second_moment = torch.reciprocal(batch_second_moment + batch_ridge)
            hwel_z = (batch_second_moment * inv_second_moment).sum()
            hwel_z_value = float(hwel_z.detach().item())
            if not math.isfinite(hwel_z_value) or hwel_z_value < 0.0:
                raise RuntimeError(
                    "Batch-online diagonal HWEL has invalid target geometry: "
                    f"normalizer_z={hwel_z_value}."
                )
            hwel_degenerate_fallback = batch_mean_second_moment.new_zeros(())
            if hwel_z_value == 0.0:
                # A fully static target batch has no data-defined HWEL scale:
                # C=0 makes Z=tr((C+ridge I)^-1 C)=0 for every positive ridge.
                # Fall back to normalized MSE instead of dividing by zero.
                # This is reported through a warning and a metric; non-finite
                # geometry raises an error above.
                if not getattr(self, "_feature_hwel_degenerate_warned", False):
                    logger.warning(
                        "Batch-online diagonal HWEL received an all-zero target delta batch "
                        "(normalizer_z=0); falling back to legacy normalized-MSE for this batch."
                    )
                    self._feature_hwel_degenerate_warned = True
                loss = normalized_mse + self.feature_cosine_weight * loss_cos
                loss_mse = normalized_mse
                hwel_nochange = batch_mean_second_moment.new_zeros(())
                hwel_gain = batch_mean_second_moment.new_zeros(())
                hwel_degenerate_fallback = batch_mean_second_moment.new_ones(())
            else:
                loss = (
                    delta_error.square() * inv_second_moment.view(1, 1, -1)
                ).sum(dim=-1).div(hwel_z).mean()
                hwel_nochange = (
                    target_delta.square() * inv_second_moment.view(1, 1, -1)
                ).sum(dim=-1).div(hwel_z).mean()
                hwel_gain = 1.0 - loss / hwel_nochange.clamp_min(1.0e-12)
                loss_mse = raw_target_mse
        else:  # pragma: no cover - config is validated earlier.
            raise RuntimeError(f"Unsupported feature loss type: {self.feature_loss_type!r}.")

        with torch.no_grad():
            pred_future = pred_future_for_grad.detach()
            pred_delta_detached = pred_delta.detach()
            endpoint_error_detached = pred_future - target_future
            delta_error_detached = pred_delta_detached - target_delta
            endpoint_mse_raw = endpoint_error_detached.square().mean()
            delta_mse_raw = delta_error_detached.square().mean()
            endpoint_rmse_raw = endpoint_mse_raw.sqrt()
            delta_rmse_raw = delta_mse_raw.sqrt()
            nochange_mse_raw = target_delta.square().mean()
            nochange_rmse_raw = nochange_mse_raw.sqrt()
            nochange_gain_raw = 1.0 - endpoint_mse_raw / nochange_mse_raw.clamp_min(1.0e-12)
            pred_target_cos = (pred_normalized.detach() * target_normalized.detach()).sum(dim=-1)
            future_recon_cos = (F.normalize(pred_future, dim=-1) * F.normalize(target_future, dim=-1)).sum(dim=-1)
            nochange_future_cos = (
                F.normalize(start_state, dim=-1) * F.normalize(target_future, dim=-1)
            ).sum(dim=-1)
            norm_ratio = pred_delta_detached.norm(dim=-1) / target_delta.norm(dim=-1).clamp_min(1.0e-12)
            metrics = {
                "loss_set_mse_raw": loss_mse.detach(),
                "loss_set_objective_raw": loss.detach(),
                "loss_set_normalized_mse": normalized_mse.detach(),
                "loss_set_raw_delta_mse": delta_mse_raw.detach(),
                "loss_set_cos_raw": loss_cos.detach(),
                "set_target_cos": pred_target_cos.mean().detach(),
                "set_future_cos": future_recon_cos.mean().detach(),
                "set_nochange_future_cos": nochange_future_cos.mean().detach(),
                "set_future_cos_gain": (future_recon_cos - nochange_future_cos).mean().detach(),
                "set_endpoint_rmse_raw": endpoint_rmse_raw.detach(),
                "set_delta_rmse_raw": delta_rmse_raw.detach(),
                "set_endpoint_delta_rmse_gap": (
                    endpoint_rmse_raw - delta_rmse_raw
                ).abs().detach(),
                "set_nochange_rmse_raw": nochange_rmse_raw.detach(),
                "set_nochange_gain_raw": nochange_gain_raw.detach(),
                "set_future_pred_norm": pred_future.norm(dim=-1).mean(),
                "set_pred_target_norm": pred_float.detach().norm(dim=-1).mean(),
                "set_target_future_norm": target_future.norm(dim=-1).mean(),
                "set_target_delta_norm": target_delta.norm(dim=-1).mean(),
                "set_pred_target_norm_ratio_median": norm_ratio.median().detach(),
                "set_pred_target_norm_ratio_q10": torch.quantile(norm_ratio, 0.1).detach(),
                "set_pred_target_norm_ratio_q90": torch.quantile(norm_ratio, 0.9).detach(),
                "set_delta_energy": target_delta.square().sum(dim=-1).mean().detach(),
                "set_feature_span": torch.as_tensor(
                    float(feature_batch["span"]), device=pred.device, dtype=pred.dtype
                ),
                "set_feature_start": torch.as_tensor(
                    float(feature_batch["start"]), device=pred.device, dtype=pred.dtype
                ),
                "set_action_start": torch.as_tensor(
                    float(feature_batch["action_start"]), device=pred.device, dtype=pred.dtype
                ),
                "set_action_end": torch.as_tensor(
                    float(feature_batch["action_end"]), device=pred.device, dtype=pred.dtype
                ),
                "set_actions_per_video_interval": torch.as_tensor(
                    float(feature_batch["actions_per_video_interval"]), device=pred.device, dtype=pred.dtype
                ),
            }
            if hwel_z is not None and hwel_nochange is not None and hwel_gain is not None:
                metrics.update(
                    {
                        "set_hwel": loss.detach(),
                        "set_hwel_nochange": hwel_nochange.detach(),
                        "set_hwel_gain": hwel_gain.detach(),
                        "set_hwel_z": hwel_z.detach(),
                    }
                )
                if self.feature_loss_type == "hwel_diagonal":
                    metrics["set_hwel_profile_z"] = hwel_z.detach()
                elif self.feature_loss_type == "hwel_diagonal_batch":
                    metrics.update(
                        {
                            "set_hwel_batch_z": hwel_z.detach(),
                            "set_hwel_batch_ridge": batch_ridge.detach(),
                            "set_hwel_batch_mean_second_moment": (
                                batch_mean_second_moment.detach()
                            ),
                            "set_hwel_degenerate_fallback": (
                                hwel_degenerate_fallback.detach()
                            ),
                        }
                    )
        return loss, metrics

    def training_loss(self, sample, tiled: bool = False):
        if not self.set_enabled:
            return super().training_loss(sample, tiled=tiled)

        profile = (
            self.history_profile_steps > 0
            and int(getattr(self, "_train_step", 0)) < self.history_profile_steps
        )
        timings: dict[str, float] = {}

        def mark(name: str, start_time: float) -> float:
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            if profile:
                timings[name] = now - start_time
            return now

        t0 = time.perf_counter()
        inputs = self.build_inputs(sample, tiled=tiled)
        t = mark("build_inputs", t0)
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        t = mark("noise_sched", t)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        t = mark("pre_dit", t)
        history_prefix = self._maybe_history_prefix(
            history_video=inputs["history_video"],
            history_action=inputs["history_action"],
            history_state=inputs["history_state"],
            tiled=tiled,
            batch_size=batch_size,
            device=action_pre["tokens"].device,
            dtype=action_pre["tokens"].dtype,
            history_dino_tokens=inputs.get("history_dino_tokens"),
            text_context=inputs.get("text_context"),
            text_context_mask=inputs.get("text_context_mask"),
        )
        t = mark("history_prefix", t)
        action_pre_with_flare = self._concat_feature_action_pre(action_pre, history_prefix)
        video_tokens = video_pre["tokens"]
        action_meta = action_pre_with_flare["meta"]
        feature_batch = self._sample_feature_prediction_batch(
            video=inputs["input_video"],
            history_frame_count=int(inputs["history_video"].shape[2]) - 1,
            action_meta=action_meta,
            current_action_len=action.shape[1],
            dtype=action_pre_with_flare["tokens"].dtype,
            device=action_pre_with_flare["tokens"].device,
            video_feature_tokens=inputs.get("video_dino_tokens"),
        )
        t = mark("feature_targets", t)
        attention_mask = self._build_history_attention_mask(
            video_seq_len=video_tokens.shape[1],
            history_prefix_len=int(action_meta["history_prefix_len"]),
            current_action_len=action.shape[1],
            extra_action_len=action.shape[1] if self.extra_prediction_action_tokens else 0,
            semantic_query_len=0,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        t = mark("attention_mask", t)
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_pre_with_flare["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre_with_flare["freqs"],
            },
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {
                    "context": action_pre_with_flare["context"],
                    "mask": action_pre_with_flare["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre_with_flare["t_mod"],
            },
        )
        t = mark("mot", t)
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        serial_action_context, serial_action_positions = self._select_serial_action_context(
            action_tokens=tokens_out["action"],
            selected_action_mask=feature_batch["selected_action_mask"],
        )
        pred_feature = self.serial_feature_predictor(
            start_feature_tokens=feature_batch["start_state"],
            action_hidden=serial_action_context,
            action_positions=serial_action_positions,
        )
        loss_set_raw, set_metrics = self._compute_feature_prediction_loss(
            pred_feature_tokens=pred_feature,
            feature_batch=feature_batch,
        )
        t = mark("feature_predictor", t)

        def post_action_group(group: str) -> torch.Tensor:
            start = int(action_meta[f"{group}_action_start"])
            end = int(action_meta[f"{group}_action_end"])
            if end <= start:
                raise ValueError(f"Action group {group!r} is empty; check SET-FLARE config.")
            return self.action_expert.post_dit(tokens_out["action"][:, start:end], action_pre)

        pred_actions = {group: post_action_group(group) for group in self.supervise_action_groups}
        pred_action = pred_actions.get(self.prediction_action_group)
        if pred_action is None:
            pred_action = post_action_group(self.prediction_action_group)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            target_action.device,
            dtype=target_action.dtype,
        )
        action_group_losses: dict[str, torch.Tensor] = {}
        for group, pred_action_group in pred_actions.items():
            action_loss_token = F.mse_loss(pred_action_group.float(), target_action.float(), reduction="none").mean(dim=2)
            if action_is_pad is not None:
                valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)
            action_group_losses[group] = (
                action_loss_per_sample * action_weight.to(action_loss_per_sample.device, action_loss_per_sample.dtype)
            ).mean()
        loss_action = torch.stack(list(action_group_losses.values())).mean()
        pred_action_clean = self._clean_action_estimate(
            noisy_action=noisy_action,
            pred_action=pred_action,
            timestep_action=timestep_action,
        )
        loss_action_psd = self._compute_history_action_psd_loss(
            history_action=inputs["history_action"],
            pred_action_clean=pred_action_clean,
            target_action=action,
            action_is_pad=action_is_pad,
        )

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.loss_lambda_action_psd * loss_action_psd
            + self.loss_lambda_set * loss_set_raw
        )
        loss_dict = {
            "loss_video": (self.loss_lambda_video * loss_video).detach(),
            "loss_action": (self.loss_lambda_action * loss_action).detach(),
            "loss_action_psd": (self.loss_lambda_action_psd * loss_action_psd).detach(),
            "loss_set": (self.loss_lambda_set * loss_set_raw).detach(),
            "loss_set_raw": loss_set_raw.detach(),
            "set_feature_tokens": torch.as_tensor(
                float(feature_batch["start_state"].shape[1]), device=loss_total.device, dtype=loss_total.dtype
            ),
            "set_action_context_tokens": torch.as_tensor(
                float(serial_action_context.shape[1]), device=loss_total.device, dtype=loss_total.dtype
            ),
            "history_prefix_tokens": torch.as_tensor(
                float(history_prefix.shape[1]), device=loss_total.device, dtype=loss_total.dtype
            ),
        }
        if getattr(self, "history_visual_memory", None) is not None:
            loss_dict["history_visual_memory_tokens"] = torch.as_tensor(
                float(self.history_visual_memory.num_queries),
                device=loss_total.device,
                dtype=loss_total.dtype,
            )
            loss_dict["history_visual_memory_frames"] = torch.as_tensor(
                float(
                    inputs["history_video"].shape[2]
                    - (0 if self.history_visual_memory_include_current else 1)
                ),
                device=loss_total.device,
                dtype=loss_total.dtype,
            )
        for name, value in set_metrics.items():
            loss_dict[name] = value.detach()
        for group, group_loss in action_group_losses.items():
            loss_dict[f"loss_action_{group}"] = (self.loss_lambda_action * group_loss).detach()
        if profile:
            mark("post_and_loss", t)
            timings["total_forward"] = time.perf_counter() - t0
            logger.info(
                "[profile:set_flare_training_loss] step=%d timings=%s shapes=%s",
                int(getattr(self, "_train_step", 0)),
                {key: round(value, 4) for key, value in timings.items()},
                {
                    "input_latents": tuple(input_latents.shape),
                    "history_video": tuple(inputs["history_video"].shape),
                    "history_prefix": tuple(history_prefix.shape),
                    "video_tokens": tuple(video_tokens.shape),
                    "action_tokens": tuple(action_pre_with_flare["tokens"].shape),
                    "feature_tokens": tuple(feature_batch["start_state"].shape),
                },
            )
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_action_noise_with_flare_cache(
        self,
        *,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        history_prefix: torch.Tensor,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre_with_flare = self._concat_feature_action_pre(action_pre, history_prefix)
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre_with_flare["tokens"],
            action_freqs=action_pre_with_flare["freqs"],
            action_t_mod=action_pre_with_flare["t_mod"],
            action_context_payload={
                "context": action_pre_with_flare["context"],
                "mask": action_pre_with_flare["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        action_meta = action_pre_with_flare["meta"]
        pred_start = int(action_meta["prediction_action_start"])
        pred_end = int(action_meta["prediction_action_end"])
        return self.action_expert.post_dit(action_tokens[:, pred_start:pred_end], action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        *args,
        history_video: Optional[torch.Tensor] = None,
        history_action: Optional[torch.Tensor] = None,
        history_state: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not self.set_enabled:
            return super().infer_action(
                *args,
                history_video=history_video,
                history_action=history_action,
                history_state=history_state,
                **kwargs,
            )
        prompt = kwargs.pop("prompt", args[0] if len(args) > 0 else None)
        input_image = kwargs.pop("input_image", args[1] if len(args) > 1 else None)
        action_horizon = kwargs.pop("action_horizon", args[2] if len(args) > 2 else None)
        if len(args) > 3:
            raise TypeError("SETFlareWAM.infer_action accepts at most positional prompt, input_image, action_horizon.")
        proprio = kwargs.pop("proprio", None)
        context = kwargs.pop("context", None)
        context_mask = kwargs.pop("context_mask", None)
        kwargs.pop("negative_prompt", None)
        kwargs.pop("text_cfg_scale", None)
        num_inference_steps = int(kwargs.pop("num_inference_steps", 20))
        sigma_shift = kwargs.pop("sigma_shift", None)
        seed = kwargs.pop("seed", None)
        rand_device = kwargs.pop("rand_device", "cpu")
        tiled = bool(kwargs.pop("tiled", False))
        if kwargs:
            raise TypeError(f"Unexpected infer_action kwargs: {sorted(kwargs)}")
        if input_image is None or action_horizon is None:
            raise ValueError("`input_image` and `action_horizon` are required.")
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'`.")

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` H/W must be multiples of 16, got {height}x{width}.")
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if prompt is None and (context is None or context_mask is None):
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        text_context = context
        text_memory = getattr(self, "history_visual_memory", None)
        text_block_index = (
            None
            if text_memory is None
            else text_memory.text_cross_attention_block_index
        )
        text_block_is_active = (
            text_block_index is not None
            and text_block_index not in text_memory.skip_block_indices
        )
        if text_block_is_active:
            text_context_mask = context_mask & (
                context.detach().float().abs().amax(dim=-1) > 0
            )
            if not bool(text_context_mask.any(dim=1).all()):
                raise ValueError(
                    "Every inference prompt must contain at least one valid text token."
                )
        else:
            text_context_mask = context_mask
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio)

        if history_video is None:
            history_video = input_image.unsqueeze(2).expand(-1, -1, self.history_video_frames, -1, -1).contiguous()
        else:
            if history_video.ndim == 4:
                history_video = history_video.unsqueeze(0)
            history_video = history_video.to(device=self.device, dtype=self.torch_dtype)
            if history_video.shape[2] == self.history_frames:
                history_video = torch.cat([history_video, input_image.unsqueeze(2)], dim=2)
        if history_action is None:
            history_horizon = self.history_action_horizon or int(action_horizon)
            history_action = torch.zeros(
                (1, history_horizon, self.action_expert.action_dim),
                device=self.device,
                dtype=self.torch_dtype,
            )
        else:
            if history_action.ndim == 2:
                history_action = history_action.unsqueeze(0)
            history_action = history_action.to(device=self.device, dtype=self.torch_dtype)

        history_state_horizon = int(history_action.shape[1]) + 1
        if history_state is None:
            if proprio is not None and proprio.shape[-1] == history_action.shape[-1]:
                history_state = proprio.unsqueeze(1).expand(-1, history_state_horizon, -1).contiguous()
            else:
                history_state = torch.zeros(
                    (1, history_state_horizon, history_action.shape[-1]),
                    device=self.device,
                    dtype=self.torch_dtype,
                )
        else:
            if history_state.ndim == 2:
                history_state = history_state.unsqueeze(0)
            history_state = history_state.to(device=self.device, dtype=self.torch_dtype)
            if history_state.shape != (1, history_state_horizon, history_action.shape[-1]):
                raise ValueError(
                    "`history_state` must be [1,history_horizon+1,action_dim], got "
                    f"{tuple(history_state.shape)}."
                )

        timestep_video = torch.zeros((first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        history_prefix = self._maybe_history_prefix(
            history_video=history_video,
            history_action=history_action,
            history_state=history_state,
            tiled=tiled,
            batch_size=1,
            device=video_pre["tokens"].device,
            dtype=video_pre["tokens"].dtype,
            text_context=text_context,
            text_context_mask=text_context_mask,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_history_attention_mask(
            video_seq_len=video_seq_len,
            history_prefix_len=int(history_prefix.shape[1]),
            current_action_len=latents_action.shape[1],
            extra_action_len=latents_action.shape[1] if self.extra_prediction_action_tokens else 0,
            semantic_query_len=0,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_flare_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                history_prefix=history_prefix,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "history_visual_tokenizer": self.history_visual_tokenizer.state_dict(),
            "history_adapter": self.history_adapter.state_dict(),
            "history_config": self.history_config,
            "action_psd_config": self.action_psd_config,
            "semantic_alignment_config": self.semantic_alignment_config,
            "set_config": self.set_config,
            "loss_lambda_action_psd": self.loss_lambda_action_psd,
            "loss_lambda_semantic_alignment": self.loss_lambda_semantic_alignment,
            "loss_lambda_set": self.loss_lambda_set,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.semantic_alignment_adapter is not None:
            payload["semantic_alignment_adapter"] = self.semantic_alignment_adapter.state_dict()
        if hasattr(self, "serial_feature_predictor"):
            payload["serial_feature_predictor"] = self.serial_feature_predictor.state_dict()
        if (
            getattr(self, "feature_loss_type", None) == "hwel_diagonal"
            and getattr(self, "feature_hwel_span_available", torch.empty(0)).numel() > 0
        ):
            payload["feature_hwel_profile"] = {
                "inv_second_moment": self.feature_hwel_inv_second_moment.detach().cpu(),
                "normalizer_z": self.feature_hwel_normalizer_z.detach().cpu(),
                "span_available": self.feature_hwel_span_available.detach().cpu(),
                "metadata": self.feature_hwel_profile_metadata,
                "sha256": self.feature_hwel_profile_sha256,
            }
        if getattr(self, "history_visual_memory", None) is not None:
            payload["history_visual_memory"] = self.history_visual_memory.state_dict()
            payload["history_visual_memory_architecture"] = (
                _history_memory_architecture_from_module(
                    self.history_config,
                    self.history_visual_memory,
                )
            )
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        checkpoint_hwel_profile = payload.get("feature_hwel_profile")
        if self.feature_loss_type == "hwel_diagonal" and checkpoint_hwel_profile is not None:
            self._set_hwel_profile_tensors(
                inv_second_moment=checkpoint_hwel_profile["inv_second_moment"],
                normalizer_z=checkpoint_hwel_profile["normalizer_z"],
                span_available=checkpoint_hwel_profile["span_available"],
                metadata=checkpoint_hwel_profile.get("metadata"),
                sha256=checkpoint_hwel_profile.get("sha256"),
            )
            logger.info(
                "Loaded diagonal HWEL profile from checkpoint %s (sha256=%s)",
                path,
                self.feature_hwel_profile_sha256,
            )
        if hasattr(self, "serial_feature_predictor"):
            if "serial_feature_predictor" in payload:
                result = self.serial_feature_predictor.load_state_dict(
                    payload["serial_feature_predictor"],
                    strict=True,
                )
                if result.missing_keys or result.unexpected_keys:
                    logger.warning(
                        "Loaded serial_feature_predictor from %s with missing=%s unexpected=%s",
                        path,
                        result.missing_keys,
                        result.unexpected_keys,
                    )
                else:
                    logger.info("Loaded serial_feature_predictor from %s", path)
            else:
                raise RuntimeError(f"Checkpoint {path} has no required `serial_feature_predictor` weights")
        if getattr(self, "history_visual_memory", None) is not None:
            if "history_visual_memory" in payload:
                current_architecture = _history_memory_architecture_from_module(
                    self.history_config,
                    self.history_visual_memory,
                )
                checkpoint_architecture = (
                    _history_memory_architecture_from_checkpoint(payload)
                )
                architecture_mismatches = (
                    _history_memory_architecture_mismatches(
                        checkpoint_architecture,
                        current_architecture,
                    )
                )
                if architecture_mismatches:
                    message = (
                        "History visual memory checkpoint/runtime architecture mismatch "
                        f"for {path}: {architecture_mismatches}. Use the checkpoint's "
                        "resolved config, or set "
                        "`history.visual_memory.allow_checkpoint_architecture_mismatch=true` "
                        "only for an intentional warm-start migration."
                    )
                    if (
                        self.history_visual_memory_allow_checkpoint_architecture_mismatch
                    ):
                        logger.warning("%s", message)
                    else:
                        raise RuntimeError(message)
                result = self.history_visual_memory.load_state_dict(
                    payload["history_visual_memory"],
                    strict=True,
                )
                if result.missing_keys or result.unexpected_keys:
                    logger.warning(
                        "Loaded history_visual_memory from %s with missing=%s unexpected=%s",
                        path,
                        result.missing_keys,
                        result.unexpected_keys,
                    )
                else:
                    logger.info("Loaded history_visual_memory from %s", path)
            else:
                raise RuntimeError(f"Checkpoint {path} has no required `history_visual_memory` weights")
        return payload


def create_aed_transition(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    history=None,
    action_psd=None,
    semantic_alignment=None,
    aed=None,
    video_lora=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    video_dit_config = _as_plain_dict(video_dit_config, "video_dit_config")
    action_dit_config = _as_plain_dict(action_dit_config, "action_dit_config")
    video_scheduler = _as_plain_dict(video_scheduler, "video_scheduler")
    action_scheduler = _as_plain_dict(action_scheduler, "action_scheduler")
    loss = _as_plain_dict(loss, "loss")
    history = _as_plain_dict(history, "history")
    action_psd = _as_plain_dict(action_psd, "action_psd")
    semantic_alignment = _as_plain_dict(semantic_alignment, "semantic_alignment")
    aed_config = _as_plain_dict(aed, "aed")
    video_lora = _as_plain_dict(video_lora, "video_lora")

    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - builtins.set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    return SETFlareWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_action_psd=float(loss.get("lambda_action_psd", 0.0)),
        loss_lambda_semantic_alignment=float(loss.get("lambda_semantic_alignment", 0.0)),
        loss_lambda_set=float(loss.get("lambda_set", loss.get("lambda_event", 0.0))),
        video_lora=video_lora,
        history_config=history,
        action_psd_config=action_psd,
        semantic_alignment_config=semantic_alignment,
        set_config=aed_config,
    )
