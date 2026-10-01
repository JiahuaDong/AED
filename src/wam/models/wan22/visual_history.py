from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # Keep lightweight utilities importable outside the training env.
    DictConfig = ()  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

from wam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .wam import WAM, _log_load_state_dict_result
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .wan_video_dit import CrossAttention, RMSNorm, sinusoidal_embedding_1d

logger = get_logger(__name__)


def _dct_basis_torch(steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"`steps` must be positive, got {steps}.")
    n = torch.arange(steps, dtype=torch.float32, device=device)
    k = torch.arange(steps, dtype=torch.float32, device=device)[:, None]
    basis = torch.cos(math.pi / float(steps) * (n + 0.5) * k)
    basis[0] *= math.sqrt(1.0 / float(steps))
    if steps > 1:
        basis[1:] *= math.sqrt(2.0 / float(steps))
    return basis.to(dtype=dtype)


def _periodogram_psd_torch(
    x: torch.Tensor,
    *,
    sample_period_frames: int,
    eps: float,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B,T,D] action tensor, got {tuple(x.shape)}")
    if sample_period_frames <= 0:
        raise ValueError(f"`sample_period_frames` must be positive, got {sample_period_frames}.")
    x = x.float()
    x = x - x.mean(dim=1, keepdim=True)
    seq_len = int(x.shape[1])
    fs = 1.0 / float(sample_period_frames)
    fft = torch.fft.rfft(x, dim=1)
    psd = fft.abs().pow(2) / (fs * max(seq_len, 1))
    if seq_len > 1:
        if seq_len % 2 == 0:
            psd[:, 1:-1, :] *= 2.0
        else:
            psd[:, 1:, :] *= 2.0
    return psd.clamp_min(eps)


def _dct_power_spectrum_torch(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B,T,D] action tensor, got {tuple(x.shape)}")
    x = x.float()
    x = x - x.mean(dim=1, keepdim=True)
    basis = _dct_basis_torch(x.shape[1], device=x.device, dtype=x.dtype)
    coeff = torch.einsum("kt,btd->bkd", basis, x)
    return coeff.pow(2).clamp_min(eps)


def _action_arm_slices(action_dim: int, *, arm_dim: int, mode: str) -> list[slice]:
    if action_dim <= 0:
        raise ValueError(f"`action_dim` must be positive, got {action_dim}.")
    if arm_dim <= 0:
        raise ValueError(f"`arm_dim` must be positive, got {arm_dim}.")

    mode = str(mode).lower()
    if mode in {"none", "single", "all"}:
        return [slice(0, action_dim)]
    if mode != "auto":
        raise ValueError(f"Unsupported action PSD arm mode: {mode}")

    num_arms = action_dim // arm_dim
    if action_dim % arm_dim == 0 and num_arms in {1, 2}:
        return [slice(i * arm_dim, (i + 1) * arm_dim) for i in range(num_arms)]
    return [slice(0, action_dim)]


def _psd_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    transform: str,
    sample_period_frames: int,
    exclude_dc: bool,
    distance: str,
    energy_mask: bool,
    energy_mask_abs_floor: float,
    energy_mask_rel_floor: float,
    eps: float,
) -> torch.Tensor:
    transform = str(transform).lower()
    if transform == "dct":
        pred_psd = _dct_power_spectrum_torch(pred, eps=eps)
        target_psd = _dct_power_spectrum_torch(target, eps=eps)
    elif transform in {"fft", "rfft", "periodogram"}:
        pred_psd = _periodogram_psd_torch(pred, sample_period_frames=sample_period_frames, eps=eps)
        target_psd = _periodogram_psd_torch(target, sample_period_frames=sample_period_frames, eps=eps)
    else:
        raise ValueError(f"Unsupported action PSD transform: {transform}")

    if exclude_dc and pred_psd.shape[1] > 1:
        pred_psd = pred_psd[:, 1:, :]
        target_psd = target_psd[:, 1:, :]

    pred_dist = pred_psd / (pred_psd.sum(dim=1, keepdim=True) + eps)
    target_dist = target_psd / (target_psd.sum(dim=1, keepdim=True) + eps)
    if distance == "sqrt":
        loss = (torch.sqrt(pred_dist + eps) - torch.sqrt(target_dist + eps)).pow(2)
    elif distance == "l1":
        loss = (pred_dist - target_dist).abs()
    elif distance == "log":
        loss = (torch.log(pred_dist + eps) - torch.log(target_dist + eps)).pow(2)
    elif distance == "mse":
        loss = (pred_dist - target_dist).pow(2)
    else:
        raise ValueError(f"Unsupported action PSD distance: {distance}")

    weight = torch.ones_like(loss)
    if energy_mask:
        target_energy = target_psd.sum(dim=1)
        rel_floor = target_energy.mean(dim=1, keepdim=True) * energy_mask_rel_floor
        energy_floor = torch.clamp(rel_floor, min=energy_mask_abs_floor)
        mask = (target_energy > energy_floor).to(dtype=loss.dtype)
        weight = weight * mask[:, None, :]
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def _as_plain_dict(value: Any, name: str, default: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if OmegaConf is not None and isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        return dict(default or {})
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must be dict-like, got {type(value)}")
    return dict(value)


class LocalFASTPlusTokenizer:
    """Safe local FAST+ encoder.

    The official FAST+ processor is a Hugging Face remote-code processor. This
    adapter reads the already-downloaded tokenizer files and mirrors the encode
    path without executing remote code.
    """

    def __init__(self, tokenizer_path: str | Path, max_tokens: int = 64) -> None:
        self.tokenizer_path = Path(tokenizer_path)
        self.max_tokens = int(max_tokens)
        if self.max_tokens <= 0:
            raise ValueError(f"`max_tokens` must be positive, got {self.max_tokens}.")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"FAST+ tokenizer path does not exist: {self.tokenizer_path}")

        try:
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            raise ImportError(
                "LocalFASTPlusTokenizer requires `transformers`. "
                "Install them or disable the history FAST+ variant."
            ) from exc

        config_path = self.tokenizer_path / "processor_config.json"
        tokenizer_file = self.tokenizer_path / "tokenizer.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing FAST+ processor config: {config_path}")
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"Missing FAST+ tokenizer file: {tokenizer_file}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.scale = float(cfg.get("scale", 10.0))
        self.vocab_size = int(cfg.get("vocab_size", 2048))
        self.min_token = int(cfg.get("min_token", 0))
        self.pad_token_id = self.vocab_size
        self._dct_basis_cache: dict[int, np.ndarray] = {}
        self._bpe_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            clean_up_tokenization_spaces=False,
        )

    def _dct_basis(self, steps: int) -> np.ndarray:
        basis = self._dct_basis_cache.get(steps)
        if basis is not None:
            return basis
        n = np.arange(steps, dtype=np.float64)
        k = np.arange(steps, dtype=np.float64)[:, None]
        basis = np.cos(np.pi / float(steps) * (n + 0.5) * k)
        basis[0] *= np.sqrt(1.0 / float(steps))
        if steps > 1:
            basis[1:] *= np.sqrt(2.0 / float(steps))
        basis = basis.astype(np.float32)
        self._dct_basis_cache[steps] = basis
        return basis

    def encode(
        self,
        actions: torch.Tensor,
        *,
        truncation_policy: str = "truncate",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim != 3:
            raise ValueError(f"`actions` must be [B,T,D], got shape {tuple(actions.shape)}")
        truncation_policy = str(truncation_policy).lower()
        if truncation_policy not in {"truncate", "error"}:
            raise ValueError(
                "`truncation_policy` must be 'truncate' or 'error', "
                f"got {truncation_policy!r}."
            )
        action_np = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        dct_coeff = np.einsum("kt,btd->bkd", self._dct_basis(action_np.shape[1]), action_np)
        dct_coeff = np.around(dct_coeff * self.scale)

        token_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []
        for row_index, elem in enumerate(dct_coeff):
            token_str = "".join(map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int)))
            ids = list(self._bpe_tokenizer(token_str)["input_ids"])
            ids = [int(x) for x in ids if 0 <= int(x) < self.vocab_size]
            if len(ids) > self.max_tokens and truncation_policy == "error":
                raise ValueError(
                    "FAST+ token sequence would be truncated: "
                    f"batch_index={row_index} encoded_length={len(ids)} max_tokens={self.max_tokens} "
                    f"for action shape [T={actions.shape[1]},D={actions.shape[2]}]. "
                    "Increase `model.history.fast_max_tokens` or use an explicitly "
                    "documented truncation policy."
                )
            ids = ids[: self.max_tokens]
            mask = [True] * len(ids)
            if len(ids) < self.max_tokens:
                pad = self.max_tokens - len(ids)
                ids.extend([self.pad_token_id] * pad)
                mask.extend([False] * pad)
            token_rows.append(ids)
            mask_rows.append(mask)

        token_ids = torch.as_tensor(token_rows, dtype=torch.long, device=actions.device)
        token_mask = torch.as_tensor(mask_rows, dtype=torch.bool, device=actions.device)
        return token_ids, token_mask


class HistoryActionVideoTransformerBlock(nn.Module):
    """Pre-norm cross-attention block used to fuse history action and video tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        ffn_mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        ffn_hidden = max(hidden_dim, int(round(hidden_dim * float(ffn_mlp_ratio))))
        self.query_norm = RMSNorm(hidden_dim, eps=eps)
        self.memory_norm = RMSNorm(hidden_dim, eps=eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.ffn_norm = RMSNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_hidden, hidden_dim),
        )
        self.out_norm = RMSNorm(hidden_dim, eps=eps)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = query + self.cross_attn(self.query_norm(query), self.memory_norm(memory), ctx_mask=ctx_mask)
        query = query + self.ffn(self.ffn_norm(query))
        return self.out_norm(query)


class HistoryLatentVisualTokenizer(nn.Module):
    """Patchify VAE history latents without running the video expert."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        patch_size: Sequence[int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = tuple(int(x) for x in patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"`patch_size` must be a 3-tuple, got {self.patch_size}.")
        self.patch_embedding = nn.Conv3d(
            self.in_dim,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = RMSNorm(self.hidden_dim, eps=eps)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(f"`latents` must be [B,C,T,H,W], got {tuple(latents.shape)}")
        if latents.shape[1] != self.in_dim:
            raise ValueError(f"`latents` channel dim must be {self.in_dim}, got {latents.shape[1]}")
        for size, patch, name in zip(latents.shape[2:], self.patch_size, ("T", "H", "W")):
            if size % patch != 0:
                raise ValueError(
                    f"History latent {name}={size} must be divisible by patch size {patch}."
                )
        x = self.patch_embedding(latents)
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return self.norm(tokens)


class HistoryActionVideoAdapter(nn.Module):
    """Build history-aware action prefix tokens from FAST+ action tokens and history video tokens."""

    def __init__(
        self,
        *,
        tokenizer_path: str | Path,
        max_action_tokens: int,
        action_hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        projector_mlp_ratio: float = 2.0,
        block_mlp_ratio: float = 4.0,
        manager_config: Optional[dict[str, Any]] = None,
        action_preprocess_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.fast_tokenizer = LocalFASTPlusTokenizer(tokenizer_path=tokenizer_path, max_tokens=max_action_tokens)
        self.max_action_tokens = int(max_action_tokens)
        self.pad_token_id = int(self.fast_tokenizer.pad_token_id)
        vocab_with_pad = int(self.fast_tokenizer.vocab_size) + 1
        preprocess_cfg = dict(action_preprocess_config or {})
        preprocess_mode = str(preprocess_cfg.get("mode", "none")).lower()
        if preprocess_mode in {"disabled", "identity"}:
            preprocess_mode = "none"
        self.action_sum_enabled = preprocess_mode in {
            "sum_by_video_frame",
            "frame_sum",
            "temporal_sum",
            "sum_by_frame",
            "random_tail_sum_by_video_frame",
        } or bool(preprocess_cfg.get("sum_by_video_frame", False))
        self.action_sum_group_size = int(preprocess_cfg.get("sum_group_size", preprocess_cfg.get("group_size", 4)))
        if self.action_sum_group_size <= 0:
            raise ValueError(
                f"`history.action_preprocess.sum_group_size` must be positive, got {self.action_sum_group_size}."
            )
        gripper_indices = preprocess_cfg.get("gripper_indices", "auto")
        self.action_sum_gripper_indices = gripper_indices
        self.action_one_token_per_group = bool(
            preprocess_cfg.get("one_token_per_group", False)
        )
        tokenization_aliases = {
            "aggregate": "aggregate_then_tokenize",
            "aggregate_t1": "aggregate_then_tokenize",
            "joint": "joint_chunk",
            "joint_tk": "joint_chunk",
        }
        group_tokenization = str(
            preprocess_cfg.get("group_tokenization", "aggregate_then_tokenize")
        ).lower()
        self.action_group_tokenization = tokenization_aliases.get(
            group_tokenization,
            group_tokenization,
        )
        if self.action_group_tokenization not in {"aggregate_then_tokenize", "joint_chunk"}:
            raise ValueError(
                "`history.action_preprocess.group_tokenization` must be "
                "'aggregate_then_tokenize' or 'joint_chunk', "
                f"got {self.action_group_tokenization!r}."
            )
        pooling_aliases = {
            "avg": "mean",
            "average": "mean",
            "attention": "positional_attention",
            "attn": "positional_attention",
        }
        group_pooling = str(preprocess_cfg.get("group_pooling", "mean")).lower()
        self.action_group_pooling = pooling_aliases.get(group_pooling, group_pooling)
        if self.action_group_pooling not in {"mean", "positional_attention"}:
            raise ValueError(
                "`history.action_preprocess.group_pooling` must be "
                "'mean' or 'positional_attention', "
                f"got {self.action_group_pooling!r}."
            )
        self.action_token_truncation_policy = str(
            preprocess_cfg.get("token_truncation_policy", "truncate")
        ).lower()
        if self.action_token_truncation_policy not in {"truncate", "error"}:
            raise ValueError(
                "`history.action_preprocess.token_truncation_policy` must be "
                "'truncate' or 'error', "
                f"got {self.action_token_truncation_policy!r}."
            )
        self.action_group_max_positions = int(preprocess_cfg.get("max_groups", 64))
        if self.action_group_max_positions <= 0:
            raise ValueError(
                "`history.action_preprocess.max_groups` must be positive, got "
                f"{self.action_group_max_positions}."
            )
        action_type = str(preprocess_cfg.get("action_type", "legacy_sum")).lower()
        action_type_aliases = {
            "abs": "absolute",
            "absolute_action": "absolute",
            "delta": "relative",
            "increment": "relative",
            "relative_action": "relative",
            "hybrid": "mixed",
            "mixed_delta": "mixed",
            "mixed_relative_absolute": "mixed",
        }
        self.action_preprocess_type = action_type_aliases.get(action_type, action_type)
        if self.action_preprocess_type not in {"absolute", "relative", "mixed", "legacy_sum"}:
            raise ValueError(
                "`history.action_preprocess.action_type` must be one of "
                "{'absolute', 'relative', 'mixed', 'legacy_sum'}, "
                f"got {action_type!r}."
            )
        self.action_relative_dim_mask = preprocess_cfg.get("relative_dim_mask", None)
        if (
            self.action_group_tokenization == "aggregate_then_tokenize"
            and self.action_preprocess_type == "mixed"
            and self.action_relative_dim_mask is None
        ):
            raise ValueError(
                "Mixed history-action aggregation requires an explicit "
                "`history.action_preprocess.relative_dim_mask`."
            )

        random_lengths = preprocess_cfg.get("random_tail_lengths", preprocess_cfg.get("random_lengths", None))
        self.action_random_tail_enabled = preprocess_mode in {
            "random_tail",
            "random_action_tail",
            "random_action_group",
            "random_tail_sum_by_video_frame",
        } or bool(preprocess_cfg.get("random_tail", False)) or bool(
            preprocess_cfg.get("random_tail_enabled", False)
        )
        if random_lengths is None:
            random_lengths = [4, 8, 12, 16, 24, 32]
        self.action_random_tail_lengths = tuple(int(x) for x in random_lengths)
        if any(length <= 0 for length in self.action_random_tail_lengths):
            raise ValueError(
                "`history.action_preprocess.random_tail_lengths` must contain positive integers, "
                f"got {self.action_random_tail_lengths}."
            )
        self.action_random_tail_eval = bool(preprocess_cfg.get("random_tail_eval", False))
        self.action_random_tail_align_video_start = bool(
            preprocess_cfg.get("random_tail_align_video_start", False)
        )
        self.action_random_tail_video_frame_ratio = preprocess_cfg.get(
            "random_tail_video_frame_ratio",
            "auto",
        )
        if (
            self.action_sum_enabled
            and self.action_group_tokenization == "aggregate_then_tokenize"
            and self.action_preprocess_type == "relative"
            and self.action_random_tail_enabled
            and not self.action_random_tail_align_video_start
        ):
            raise ValueError(
                "Relative grouped random-tail actions require "
                "`history.action_preprocess.random_tail_align_video_start=true` so the window start state can move."
            )
        if self.action_one_token_per_group and not self.action_sum_enabled:
            raise ValueError(
                "`history.action_preprocess.one_token_per_group=true` requires grouped action preprocessing."
            )
        if self.action_group_tokenization == "joint_chunk" and not self.action_one_token_per_group:
            raise ValueError(
                "`history.action_preprocess.group_tokenization=joint_chunk` requires "
                "`one_token_per_group=true`."
            )
        if self.action_group_pooling == "positional_attention" and not self.action_one_token_per_group:
            raise ValueError(
                "`history.action_preprocess.group_pooling=positional_attention` requires "
                "`one_token_per_group=true`."
            )

        hidden = max(action_hidden_dim, int(round(action_hidden_dim * float(projector_mlp_ratio))))
        self.action_token_embedding = nn.Embedding(vocab_with_pad, action_hidden_dim, padding_idx=self.pad_token_id)
        self.action_token_projector = nn.Sequential(
            nn.LayerNorm(action_hidden_dim, eps=eps),
            nn.Linear(action_hidden_dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim, eps=eps),
        )
        if self.action_one_token_per_group:
            self.action_group_pos = nn.Parameter(
                torch.zeros(1, self.action_group_max_positions, action_hidden_dim)
            )
            nn.init.trunc_normal_(self.action_group_pos, std=0.02)
        else:
            self.register_parameter("action_group_pos", None)
        if self.action_group_pooling == "positional_attention":
            self.action_token_pool_pos = nn.Parameter(
                torch.zeros(1, self.max_action_tokens, action_hidden_dim)
            )
            self.action_token_pool_query = nn.Parameter(
                torch.zeros(1, 1, action_hidden_dim)
            )
            nn.init.trunc_normal_(self.action_token_pool_pos, std=0.02)
            nn.init.trunc_normal_(self.action_token_pool_query, std=0.02)
            self.action_token_pool_norm = nn.LayerNorm(action_hidden_dim, eps=eps)
            self.action_token_pool_out_norm = nn.LayerNorm(action_hidden_dim, eps=eps)
        else:
            self.register_parameter("action_token_pool_pos", None)
            self.register_parameter("action_token_pool_query", None)
            self.action_token_pool_norm = None
            self.action_token_pool_out_norm = None
        self.transformer_block = HistoryActionVideoTransformerBlock(
            hidden_dim=action_hidden_dim,
            attn_head_dim=attn_head_dim,
            num_heads=num_heads,
            eps=eps,
            ffn_mlp_ratio=block_mlp_ratio,
        )
        manager_cfg = dict(manager_config or {})
        manager_enabled = bool(manager_cfg.get("enabled", False))
        manager_type = str(manager_cfg.get("type", "identity")).lower()
        if manager_enabled and manager_type not in {"identity", "none", "disabled"}:
            raise ValueError(
                "`history.manager.type` only supports 'identity', "
                f"got {manager_type!r}."
            )
        else:
            self.history_manager = nn.Identity()
            self.history_manager_type = "identity"

    @staticmethod
    def _auto_gripper_indices(action_dim: int) -> list[int]:
        if action_dim % 7 == 0 and action_dim // 7 in {1, 2}:
            return [7 * arm + 6 for arm in range(action_dim // 7)]
        if action_dim % 8 == 0 and action_dim // 8 in {1, 2}:
            return [8 * arm + 7 for arm in range(action_dim // 8)]
        return []

    def _resolve_gripper_indices(self, action_dim: int, device: torch.device) -> torch.Tensor:
        raw = self.action_sum_gripper_indices
        if raw is None or str(raw).lower() in {"none", "false", "off", "disabled"}:
            indices: list[int] = []
        elif isinstance(raw, str) and raw.lower() == "auto":
            indices = self._auto_gripper_indices(action_dim)
        else:
            indices = [int(x) for x in raw]
        indices = [idx for idx in indices if 0 <= idx < action_dim]
        return torch.as_tensor(sorted(set(indices)), dtype=torch.long, device=device)

    def _resolve_relative_dim_mask(self, action_dim: int, device: torch.device) -> Optional[torch.Tensor]:
        raw = self.action_relative_dim_mask
        if raw is None:
            return None
        mask = torch.as_tensor(raw, dtype=torch.bool, device=device)
        if mask.ndim != 1 or mask.numel() != action_dim:
            raise ValueError(
                "`history.action_preprocess.relative_dim_mask` must contain one boolean per action "
                f"dimension, got shape {tuple(mask.shape)} for action_dim={action_dim}."
            )
        if not bool(mask.any()):
            raise ValueError("Mixed history-action aggregation requires at least one relative dimension.")
        if bool(mask.all()):
            raise ValueError("Mixed history-action aggregation requires at least one absolute dimension.")
        return mask

    def _maybe_random_tail(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.action_random_tail_enabled:
            return actions
        if self.action_random_tail_align_video_start:
            return actions
        if not self.training and not self.action_random_tail_eval:
            return actions
        horizon = int(actions.shape[1])
        valid_lengths = [length for length in self.action_random_tail_lengths if length <= horizon]
        if not valid_lengths:
            return actions
        length = random.choice(valid_lengths)
        return actions[:, -length:]

    @staticmethod
    def _aggregate_action_groups(
        actions: torch.Tensor,
        *,
        group_size: int,
        action_type: str,
        history_start_state: Optional[torch.Tensor] = None,
        absolute_dim_indices: Optional[torch.Tensor] = None,
        relative_dim_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert action-rate commands into fixed-size video-level endpoints."""
        if actions.ndim != 3:
            raise ValueError(f"`actions` must be [B,T,D], got {tuple(actions.shape)}")
        if group_size <= 0:
            raise ValueError(f"`group_size` must be positive, got {group_size}.")
        action_type = str(action_type).lower()
        if action_type not in {"absolute", "relative", "mixed", "legacy_sum"}:
            raise ValueError(
                "`action_type` must be 'absolute', 'relative', 'mixed', or 'legacy_sum', "
                f"got {action_type!r}."
            )

        batch, horizon, action_dim = actions.shape
        remainder = horizon % group_size
        if remainder:
            pad_len = group_size - remainder
            pad = actions.new_zeros((batch, pad_len, action_dim))
            actions = torch.cat([pad, actions], dim=1)
            horizon = int(actions.shape[1])
        grouped = actions.view(batch, horizon // group_size, group_size, action_dim)
        if action_type == "absolute":
            return grouped[:, :, -1, :].contiguous()

        if action_type == "legacy_sum":
            summed = grouped.sum(dim=2)
            if absolute_dim_indices is not None and absolute_dim_indices.numel() > 0:
                absolute_dim_indices = absolute_dim_indices.to(device=actions.device, dtype=torch.long)
                summed[:, :, absolute_dim_indices] = grouped[:, :, -1, absolute_dim_indices]
            return summed

        if action_type == "mixed":
            if relative_dim_mask is None:
                raise ValueError("Mixed action aggregation requires `relative_dim_mask`.")
            relative_dim_mask = relative_dim_mask.to(device=actions.device, dtype=torch.bool)
            if relative_dim_mask.ndim != 1 or relative_dim_mask.numel() != action_dim:
                raise ValueError(
                    "`relative_dim_mask` must be [action_dim], got "
                    f"{tuple(relative_dim_mask.shape)} for action_dim={action_dim}."
                )
            aggregated = grouped[:, :, -1, :].clone()
            aggregated[:, :, relative_dim_mask] = grouped[:, :, :, relative_dim_mask].sum(dim=2)
            return aggregated

        if history_start_state is None:
            raise ValueError(
                "Relative history-action aggregation requires `history_start_state` for the selected window."
            )
        if history_start_state.ndim == 3 and history_start_state.shape[1] == 1:
            history_start_state = history_start_state[:, 0]
        if history_start_state.shape != (batch, action_dim):
            raise ValueError(
                "`history_start_state` must match relative action shape [B,D], got "
                f"{tuple(history_start_state.shape)} for actions {tuple(actions.shape)}."
            )
        history_start_state = history_start_state.to(device=actions.device, dtype=actions.dtype)

        grouped_delta = grouped.sum(dim=2)
        endpoints = history_start_state.unsqueeze(1) + grouped_delta.cumsum(dim=1)
        if absolute_dim_indices is not None and absolute_dim_indices.numel() > 0:
            absolute_dim_indices = absolute_dim_indices.to(device=actions.device, dtype=torch.long)
            endpoints[:, :, absolute_dim_indices] = grouped[:, :, -1, absolute_dim_indices]
        return endpoints

    def _sum_by_video_frame(
        self,
        actions: torch.Tensor,
        history_start_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.action_sum_enabled:
            return actions
        if self.action_one_token_per_group and actions.shape[1] % self.action_sum_group_size != 0:
            raise ValueError(
                "One-token-per-group history requires an exact temporal partition, got "
                f"{actions.shape[1]} actions with group size {self.action_sum_group_size}."
            )
        absolute_indices = self._resolve_gripper_indices(actions.shape[-1], actions.device)
        relative_dim_mask = self._resolve_relative_dim_mask(actions.shape[-1], actions.device)
        return self._aggregate_action_groups(
            actions,
            group_size=self.action_sum_group_size,
            action_type=self.action_preprocess_type,
            history_start_state=history_start_state,
            absolute_dim_indices=absolute_indices,
            relative_dim_mask=relative_dim_mask,
        )

    def _preprocess_history_action(
        self,
        history_action: torch.Tensor,
        history_start_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        actions = self._maybe_random_tail(history_action)
        if self.action_group_tokenization == "joint_chunk":
            return actions
        actions = self._sum_by_video_frame(actions, history_start_state=history_start_state)
        return actions

    def _pool_group_token_hidden(
        self,
        token_hidden: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_hidden.ndim != 3:
            raise ValueError(
                f"`token_hidden` must be [B,L,D], got {tuple(token_hidden.shape)}"
            )
        if token_mask.shape != token_hidden.shape[:2]:
            raise ValueError(
                "`token_mask` must match token_hidden [B,L], got "
                f"{tuple(token_mask.shape)} for {tuple(token_hidden.shape)}."
            )
        if self.action_group_pooling == "mean":
            denominator = token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(
                dtype=token_hidden.dtype
            )
            return token_hidden.sum(dim=1) / denominator

        if not torch.all(token_mask.any(dim=1)):
            raise ValueError(
                "Positional attention pooling requires at least one valid FAST+ token per action group."
            )
        token_count = int(token_hidden.shape[1])
        if token_count > self.max_action_tokens:
            raise ValueError(
                f"FAST+ token count {token_count} exceeds max_tokens={self.max_action_tokens}."
            )
        if (
            self.action_token_pool_pos is None
            or self.action_token_pool_query is None
            or self.action_token_pool_norm is None
            or self.action_token_pool_out_norm is None
        ):
            raise RuntimeError("Positional attention pooling modules were not initialized.")
        pool_input = token_hidden + self.action_token_pool_pos[:, :token_count].to(
            device=token_hidden.device,
            dtype=token_hidden.dtype,
        )
        pool_key = self.action_token_pool_norm(pool_input)
        pool_query = self.action_token_pool_query.to(
            device=token_hidden.device,
            dtype=token_hidden.dtype,
        )
        logits = (pool_key * pool_query).sum(dim=-1) / math.sqrt(float(token_hidden.shape[-1]))
        logits = logits.masked_fill(~token_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled = (weights.unsqueeze(-1) * pool_input).sum(dim=1)
        return self.action_token_pool_out_norm(pooled)

    def forward(
        self,
        history_action: torch.Tensor,
        history_video_tokens: torch.Tensor,
        history_start_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_video_tokens.ndim != 3:
            raise ValueError(
                f"`history_video_tokens` must be [B,N,D], got {tuple(history_video_tokens.shape)}"
            )
        query, token_mask = self.encode_action_tokens(
            history_action,
            history_start_state=history_start_state,
            device=history_video_tokens.device,
        )
        aware = self.transformer_block(query, history_video_tokens)
        aware = self.history_manager(aware)
        aware = aware * token_mask.unsqueeze(-1).to(dtype=aware.dtype)
        return aware, token_mask

    def encode_action_tokens(
        self,
        history_action: torch.Tensor,
        *,
        history_start_state: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess and encode history actions without applying visual cross-attention."""
        history_action = self._preprocess_history_action(
            history_action,
            history_start_state=history_start_state,
        )
        if self.action_one_token_per_group:
            if self.action_group_tokenization == "joint_chunk":
                batch, horizon, action_dim = history_action.shape
                if horizon % self.action_sum_group_size != 0:
                    raise ValueError(
                        "One-token-per-group history requires an exact temporal partition, got "
                        f"{horizon} actions with group size {self.action_sum_group_size}."
                    )
                num_groups = horizon // self.action_sum_group_size
                flat_actions = history_action.reshape(
                    batch * num_groups,
                    self.action_sum_group_size,
                    action_dim,
                )
            else:
                batch, num_groups, action_dim = history_action.shape
                flat_actions = history_action.reshape(batch * num_groups, 1, action_dim)
            token_ids, token_mask = self.fast_tokenizer.encode(
                flat_actions,
                truncation_policy=self.action_token_truncation_policy,
            )
            target_device = history_action.device if device is None else device
            token_ids = token_ids.to(device=target_device)
            token_mask = token_mask.to(device=target_device)
            token_hidden = self.action_token_embedding(token_ids)
            token_hidden = token_hidden * token_mask.unsqueeze(-1).to(dtype=token_hidden.dtype)
            token_hidden = self.action_token_projector(token_hidden)
            token_hidden = token_hidden * token_mask.unsqueeze(-1).to(dtype=token_hidden.dtype)
            grouped_hidden = self._pool_group_token_hidden(token_hidden, token_mask)
            grouped_hidden = grouped_hidden.reshape(batch, num_groups, -1)
            if num_groups > self.action_group_max_positions:
                raise ValueError(
                    f"History action groups {num_groups} exceed max_groups={self.action_group_max_positions}."
                )
            group_pos_start = self.action_group_max_positions - num_groups
            grouped_hidden = grouped_hidden + self.action_group_pos[:, group_pos_start:].to(
                device=grouped_hidden.device,
                dtype=grouped_hidden.dtype,
            )
            group_mask = torch.ones(
                (batch, num_groups),
                dtype=torch.bool,
                device=target_device,
            )
            return grouped_hidden, group_mask

        token_ids, token_mask = self.fast_tokenizer.encode(history_action)
        target_device = history_action.device if device is None else device
        token_ids = token_ids.to(device=target_device)
        token_mask = token_mask.to(device=target_device)

        query = self.action_token_embedding(token_ids)
        query = query * token_mask.unsqueeze(-1).to(dtype=query.dtype)
        query = self.action_token_projector(query)
        query = query * token_mask.unsqueeze(-1).to(dtype=query.dtype)
        return query, token_mask


class SemanticAlignmentAdapter(nn.Module):
    """Learn a semantic action query and regress it to pooled text context."""

    def __init__(
        self,
        *,
        action_dim: int,
        text_dim: int,
        num_queries: int = 1,
        projector_mlp_ratio: float = 1.0,
        normalize_target: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.text_dim = int(text_dim)
        self.num_queries = int(num_queries)
        self.normalize_target = bool(normalize_target)
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {action_dim}.")
        if self.text_dim <= 0:
            raise ValueError(f"`text_dim` must be positive, got {text_dim}.")
        if self.num_queries <= 0:
            raise ValueError(f"`num_queries` must be positive, got {num_queries}.")
        self.query = nn.Parameter(
            torch.randn(1, self.num_queries, self.action_dim, dtype=torch.float32) * 0.02
        )
        hidden = max(self.text_dim, int(round(self.text_dim * float(projector_mlp_ratio))))
        self.query_to_text = nn.Sequential(
            nn.LayerNorm(self.action_dim, eps=float(eps)),
            nn.Linear(self.action_dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, self.text_dim),
            nn.LayerNorm(self.text_dim, eps=float(eps)),
        )
        self.text_norm = nn.LayerNorm(self.text_dim, eps=float(eps))

    def query_tokens(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.query.to(device=device, dtype=dtype).expand(batch_size, -1, -1).contiguous()

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        if mask.ndim != 2:
            raise ValueError(f"`mask` must be [B,L], got {tuple(mask.shape)}")
        valid = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def text_target(self, context: torch.Tensor, context_mask: Optional[torch.Tensor]) -> torch.Tensor:
        text = self.text_norm(context.to(dtype=self.text_norm.weight.dtype))
        target = self._masked_mean(text, context_mask).float()
        if self.normalize_target:
            target = F.normalize(target, dim=-1)
        return target

    def forward(
        self,
        *,
        query_hidden: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query_hidden.ndim != 3 or context.ndim != 3:
            raise ValueError(
                "Semantic query alignment expects query/context tensors shaped [B,N,D]. "
                f"Got query={tuple(query_hidden.shape)}, context={tuple(context.shape)}."
            )
        query_feat = self.query_to_text(query_hidden).mean(dim=1).float()
        target = self.text_target(context, context_mask).to(device=query_feat.device)
        if self.normalize_target:
            query_feat = F.normalize(query_feat, dim=-1)
        loss = F.mse_loss(query_feat, target, reduction="mean")
        with torch.no_grad():
            metrics = {
                "semantic_alignment_query_text_mse": loss.detach(),
                "semantic_alignment_query_text_cos": (
                    F.normalize(query_feat, dim=-1) * F.normalize(target, dim=-1)
                ).sum(dim=-1).mean(),
                "semantic_alignment_query_norm": query_feat.norm(dim=-1).mean(),
                "semantic_alignment_text_norm": target.norm(dim=-1).mean(),
            }
        return loss, metrics


class HistoryAwareWAM(WAM):
    """WAM variant with FAST+ history-action prefix tokens.

    This class intentionally lives outside the original WAM implementation.
    If a batch provides `history_video` and `history_action`, those fields are
    used directly. Otherwise the history branch follows the requested boundary
    behavior: repeat the current frame and use zero history actions.
    """

    def __init__(
        self,
        *args,
        history_config: Optional[dict[str, Any]] = None,
        action_psd_config: Optional[dict[str, Any]] = None,
        semantic_alignment_config: Optional[dict[str, Any]] = None,
        loss_lambda_action_psd: float = 0.0,
        loss_lambda_semantic_alignment: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        config = dict(history_config or {})
        psd_config = dict(action_psd_config or {})
        semantic_config = dict(semantic_alignment_config or {})
        self.history_config = config
        self.action_psd_config = psd_config
        self.semantic_alignment_config = semantic_config
        self.semantic_alignment_target_context = str(semantic_config.get("target_context", "task")).lower()
        if self.semantic_alignment_target_context not in {"task", "prompt"}:
            raise ValueError(
                "`semantic_alignment.target_context` must be either 'task' or 'prompt', "
                f"got {self.semantic_alignment_target_context!r}."
            )
        self.semantic_alignment_fallback_to_prompt = bool(semantic_config.get("fallback_to_prompt", True))
        self.history_frames = int(config.get("history_frames", 8))
        self.history_video_frames = self.history_frames + 1
        self.history_action_horizon = config.get("history_action_horizon", None)
        if self.history_action_horizon is not None:
            self.history_action_horizon = int(self.history_action_horizon)
        self.history_fast_max_tokens = int(config.get("fast_max_tokens", 64))
        self.history_prefix_attend_current = bool(config.get("prefix_attend_current_action", False))
        self.history_guidance_scale = float(
            os.environ.get("WAM_HISTORY_GUIDANCE_SCALE", config.get("guidance_scale", 1.0))
        )
        if not math.isfinite(self.history_guidance_scale):
            raise ValueError(
                "`history.guidance_scale` must be finite, "
                f"got {self.history_guidance_scale}."
            )
        self.history_train_dropout_prob = float(
            config.get("train_dropout_prob", config.get("history_dropout_prob", 0.0))
        )
        if not (0.0 <= self.history_train_dropout_prob <= 1.0):
            raise ValueError(
                "`history.train_dropout_prob` must be in [0,1], "
                f"got {self.history_train_dropout_prob}."
            )
        self.history_train_dropout_mode = str(config.get("train_dropout_mode", "zero_all")).lower()
        valid_dropout_modes = {"zero_all", "video_only", "action_only"}
        if self.history_train_dropout_mode not in valid_dropout_modes:
            raise ValueError(
                "`history.train_dropout_mode` must be one of "
                f"{sorted(valid_dropout_modes)}, got {self.history_train_dropout_mode!r}."
            )
        self.extra_prediction_action_tokens = bool(config.get("extra_prediction_action_tokens", False))
        self.extra_prediction_attend_history = bool(config.get("extra_prediction_attend_history", False))
        default_prediction_group = "extra" if self.extra_prediction_action_tokens else "current"
        self.prediction_action_group = str(config.get("prediction_action_group", default_prediction_group)).lower()
        if self.prediction_action_group not in {"current", "extra"}:
            raise ValueError(
                "`history.prediction_action_group` must be 'current' or 'extra', "
                f"got {self.prediction_action_group!r}."
            )
        if self.prediction_action_group == "extra" and not self.extra_prediction_action_tokens:
            raise ValueError(
                "`history.prediction_action_group='extra'` requires "
                "`history.extra_prediction_action_tokens=true`."
            )
        supervise_groups_raw = config.get("supervise_action_groups", self.prediction_action_group)
        if isinstance(supervise_groups_raw, str):
            supervise_text = supervise_groups_raw.strip().lower()
            if supervise_text in {"both", "all"}:
                supervise_groups = ["current", "extra"]
            else:
                supervise_groups = [part.strip().lower() for part in supervise_text.split(",") if part.strip()]
        else:
            supervise_groups = [str(group).strip().lower() for group in supervise_groups_raw]
        if not supervise_groups:
            raise ValueError("`history.supervise_action_groups` must supervise at least one action group.")
        invalid_supervise_groups = sorted(set(supervise_groups) - {"current", "extra"})
        if invalid_supervise_groups:
            raise ValueError(
                "`history.supervise_action_groups` only supports 'current', 'extra', or 'both', "
                f"got {invalid_supervise_groups}."
            )
        if "extra" in supervise_groups and not self.extra_prediction_action_tokens:
            raise ValueError(
                "`history.supervise_action_groups` includes 'extra' but "
                "`history.extra_prediction_action_tokens=false`."
            )
        self.supervise_action_groups = tuple(dict.fromkeys(supervise_groups))
        self.loss_lambda_action_psd = float(loss_lambda_action_psd)
        self.loss_lambda_semantic_alignment = float(loss_lambda_semantic_alignment)
        self.action_psd_transform = str(psd_config.get("transform", "dct")).lower()
        self.action_psd_distance = str(psd_config.get("distance", "sqrt")).lower()
        self.action_psd_sample_period_frames = int(psd_config.get("sample_period_frames", 1))
        self.action_psd_exclude_dc = bool(psd_config.get("exclude_dc", True))
        self.action_psd_energy_mask = bool(psd_config.get("energy_mask", True))
        self.action_psd_energy_mask_abs_floor = float(psd_config.get("energy_mask_abs_floor", 1e-8))
        self.action_psd_energy_mask_rel_floor = float(psd_config.get("energy_mask_rel_floor", 1e-3))
        self.action_psd_arm_dim = int(psd_config.get("arm_dim", 7))
        self.action_psd_arm_mode = str(psd_config.get("arm_mode", "auto")).lower()
        self.action_psd_eps = float(psd_config.get("eps", 1e-8))
        profile_steps_default = os.environ.get("WAM_PROFILE_STEPS", "0")
        self.history_profile_steps = int(
            os.environ.get("WAM_HISTORY_PROFILE_STEPS", profile_steps_default)
        )

        tokenizer_path = config.get("fast_tokenizer_path", "third_party/fast_plus_tokenizer")
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.is_absolute():
            tokenizer_path = Path(__file__).resolve().parents[4] / tokenizer_path

        action_hidden_dim = int(self.action_expert.hidden_dim)
        video_latent_dim = int(getattr(self.video_expert, "in_dim"))
        eps = float(config.get("eps", getattr(self.action_expert.blocks[0], "norm1").eps))
        self.history_visual_tokenizer = HistoryLatentVisualTokenizer(
            in_dim=video_latent_dim,
            hidden_dim=action_hidden_dim,
            patch_size=getattr(self.video_expert, "patch_size"),
            eps=eps,
        )
        self.history_adapter = HistoryActionVideoAdapter(
            tokenizer_path=tokenizer_path,
            max_action_tokens=self.history_fast_max_tokens,
            action_hidden_dim=action_hidden_dim,
            attn_head_dim=int(self.action_expert.attn_head_dim),
            num_heads=int(self.action_expert.num_heads),
            eps=eps,
            projector_mlp_ratio=float(config.get("projector_mlp_ratio", 2.0)),
            block_mlp_ratio=float(config.get("block_mlp_ratio", 4.0)),
            manager_config=config.get("manager", None),
            action_preprocess_config=config.get("action_preprocess", None),
        )
        self.semantic_alignment_enabled = bool(semantic_config.get("enabled", False)) and self.loss_lambda_semantic_alignment > 0.0
        if self.semantic_alignment_enabled:
            self.semantic_alignment_adapter = SemanticAlignmentAdapter(
                action_dim=int(self.action_expert.hidden_dim),
                text_dim=int(self.text_dim),
                num_queries=int(semantic_config.get("num_queries", 1)),
                projector_mlp_ratio=float(semantic_config.get("projector_mlp_ratio", 1.0)),
                normalize_target=bool(semantic_config.get("normalize_target", False)),
                eps=eps,
            )
        else:
            self.semantic_alignment_adapter = None
        self.history_visual_tokenizer.to(device=self.device, dtype=self.torch_dtype)
        self.history_adapter.to(device=self.device, dtype=self.torch_dtype)
        if self.semantic_alignment_adapter is not None:
            self.semantic_alignment_adapter.to(device=self.device, dtype=self.torch_dtype)
        self._train_step = 0
        logger.info("History timing profile config: history_profile_steps=%d", self.history_profile_steps)
        logger.info(
            "History dual-action config: extra_prediction_action_tokens=%s prediction_group=%s supervise_groups=%s extra_attend_history=%s",
            self.extra_prediction_action_tokens,
            self.prediction_action_group,
            ",".join(self.supervise_action_groups),
            self.extra_prediction_attend_history,
        )
        logger.info(
            "History action PSD loss config: weight=%.4g transform=%s distance=%s arm_mode=%s arm_dim=%d",
            self.loss_lambda_action_psd,
            self.action_psd_transform,
            self.action_psd_distance,
            self.action_psd_arm_mode,
            self.action_psd_arm_dim,
        )
        logger.info(
            "History manager config: type=%s fast_tokens=%d",
            getattr(self.history_adapter, "history_manager_type", "unknown"),
            self.history_fast_max_tokens,
        )
        logger.info(
            "History action preprocess: sum_by_video_frame=%s action_type=%s sum_group=%d "
            "one_token_per_group=%s group_tokenization=%s group_pooling=%s "
            "token_truncation_policy=%s random_tail=%s random_lengths=%s",
            getattr(self.history_adapter, "action_sum_enabled", False),
            getattr(self.history_adapter, "action_preprocess_type", "legacy_sum"),
            getattr(self.history_adapter, "action_sum_group_size", 4),
            getattr(self.history_adapter, "action_one_token_per_group", False),
            getattr(self.history_adapter, "action_group_tokenization", "aggregate_then_tokenize"),
            getattr(self.history_adapter, "action_group_pooling", "mean"),
            getattr(self.history_adapter, "action_token_truncation_policy", "truncate"),
            getattr(self.history_adapter, "action_random_tail_enabled", False),
            getattr(self.history_adapter, "action_random_tail_lengths", ()),
        )
        logger.info(
            "History train dropout config: prob=%.3g mode=%s",
            self.history_train_dropout_prob,
            self.history_train_dropout_mode,
        )
        logger.info(
            "History inference guidance config: scale=%.4g",
            self.history_guidance_scale,
        )
        logger.info(
            "History semantic alignment config: enabled=%s weight=%.4g num_queries=%d",
            self.semantic_alignment_enabled,
            self.loss_lambda_semantic_alignment,
            int(semantic_config.get("num_queries", 1)),
        )

    def set_train_step(self, step: int):
        self._train_step = int(step)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "history_visual_tokenizer"):
            self.history_visual_tokenizer.to(*args, **kwargs)
        if hasattr(self, "history_adapter"):
            self.history_adapter.to(*args, **kwargs)
        if getattr(self, "semantic_alignment_adapter", None) is not None:
            self.semantic_alignment_adapter.to(*args, **kwargs)
        return self

    def extra_trainable_modules(self):
        modules = [self.history_visual_tokenizer, self.history_adapter]
        if getattr(self, "semantic_alignment_adapter", None) is not None:
            modules.append(self.semantic_alignment_adapter)
        return modules

    def extra_trainable_parameters(self):
        params = list(self.history_visual_tokenizer.parameters()) + list(self.history_adapter.parameters())
        if getattr(self, "semantic_alignment_adapter", None) is not None:
            params.extend(self.semantic_alignment_adapter.parameters())
        return params

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_action_psd: float = 0.0,
        loss_lambda_semantic_alignment: float = 0.0,
        video_lora: dict[str, Any] | None = None,
        history_config: Optional[dict[str, Any]] = None,
        action_psd_config: Optional[dict[str, Any]] = None,
        semantic_alignment_config: Optional[dict[str, Any]] = None,
    ) -> "HistoryAwareWAM":
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for HistoryAwareWAM.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for HistoryAwareWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_action_psd=loss_lambda_action_psd,
            loss_lambda_semantic_alignment=loss_lambda_semantic_alignment,
            history_config=history_config,
            action_psd_config=action_psd_config,
            semantic_alignment_config=semantic_alignment_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "fast_plus_tokenizer": str(model.history_adapter.fast_tokenizer.tokenizer_path),
            "history_manager": getattr(model.history_adapter, "history_manager_type", "identity"),
        }
        model.configure_video_lora(video_lora)
        return model

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        semantic_context = sample.get("semantic_context", None)
        semantic_context_mask = sample.get("semantic_context_mask", None)
        if semantic_context is not None or semantic_context_mask is not None:
            if semantic_context is None or semantic_context_mask is None:
                raise ValueError("`semantic_context` and `semantic_context_mask` must be provided together.")
            if semantic_context.ndim != 3 or semantic_context_mask.ndim != 2:
                raise ValueError(
                    "`semantic_context/semantic_context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(semantic_context.shape)} and {tuple(semantic_context_mask.shape)}"
                )
            inputs["semantic_context"] = semantic_context.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            inputs["semantic_context_mask"] = semantic_context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        video = inputs["input_video"]
        current_frame = video[:, :, 0:1]

        history_video = sample.get("history_video", None)
        if history_video is None:
            history_video = current_frame.expand(-1, -1, self.history_video_frames, -1, -1).contiguous()
        else:
            history_video = history_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            if history_video.ndim != 5:
                raise ValueError(f"`history_video` must be [B,3,T,H,W], got {tuple(history_video.shape)}")
            if history_video.shape[2] == self.history_frames:
                history_video = torch.cat([history_video, current_frame], dim=2)
            elif history_video.shape[2] != self.history_video_frames:
                raise ValueError(
                    "`history_video` must contain either history frames only or history+current frames: "
                    f"got T={history_video.shape[2]}, expected {self.history_frames} or {self.history_video_frames}."
                )

        action = inputs["action"]
        history_horizon = self.history_action_horizon or int(action.shape[1])
        history_action = sample.get("history_action", None)
        if history_action is None:
            history_action = torch.zeros(
                action.shape[0],
                history_horizon,
                action.shape[2],
                device=self.device,
                dtype=action.dtype,
            )
        else:
            history_action = history_action.to(device=self.device, dtype=action.dtype, non_blocking=True)
            if history_action.ndim != 3 or history_action.shape[0] != action.shape[0] or history_action.shape[2] != action.shape[2]:
                raise ValueError(
                    "`history_action` must be [B,T,action_dim], got "
                    f"{tuple(history_action.shape)} for action {tuple(action.shape)}"
                )
            if history_action.shape[1] < history_horizon:
                pad = torch.zeros(
                    history_action.shape[0],
                    history_horizon - history_action.shape[1],
                    history_action.shape[2],
                    device=history_action.device,
                    dtype=history_action.dtype,
                )
                history_action = torch.cat([pad, history_action], dim=1)
            elif history_action.shape[1] > history_horizon:
                history_action = history_action[:, -history_horizon:]

        history_state = sample.get("history_state", None)
        requires_history_start = bool(
            self.history_adapter.action_sum_enabled
            and self.history_adapter.action_group_tokenization == "aggregate_then_tokenize"
            and self.history_adapter.action_preprocess_type == "relative"
        )
        if history_state is None:
            if requires_history_start:
                raise ValueError(
                    "Relative history-action aggregation requires `history_state` from the dataset."
                )
            history_state = torch.zeros(
                action.shape[0],
                history_horizon + 1,
                action.shape[2],
                device=self.device,
                dtype=action.dtype,
            )
        else:
            history_state = history_state.to(device=self.device, dtype=action.dtype, non_blocking=True)
            if history_state.ndim != 3 or history_state.shape[0] != action.shape[0]:
                raise ValueError(
                    "`history_state` must be [B,T+1,action_dim], got "
                    f"{tuple(history_state.shape)} for action {tuple(action.shape)}"
                )
            if history_state.shape[2] != action.shape[2]:
                if requires_history_start:
                    raise ValueError(
                        "Relative history-action aggregation requires state/action dimensions to match, got "
                        f"{history_state.shape[2]} and {action.shape[2]}."
                    )
                history_state = torch.zeros(
                    action.shape[0],
                    history_horizon + 1,
                    action.shape[2],
                    device=self.device,
                    dtype=action.dtype,
                )
            target_state_horizon = history_horizon + 1
            if history_state.shape[1] < target_state_horizon:
                pad_state = history_state[:, :1].expand(
                    -1,
                    target_state_horizon - history_state.shape[1],
                    -1,
                )
                history_state = torch.cat([pad_state, history_state], dim=1)
            elif history_state.shape[1] > target_state_horizon:
                history_state = history_state[:, -target_state_horizon:]

        if self.training and self.history_train_dropout_prob > 0.0:
            drop_mask = torch.rand(
                (action.shape[0],),
                device=self.device,
                dtype=torch.float32,
            ) < self.history_train_dropout_prob
            if self.history_train_dropout_mode in {"zero_all", "action_only"}:
                zero_action = torch.zeros_like(history_action)
                history_action = torch.where(drop_mask[:, None, None], zero_action, history_action)
                current_state = history_state[:, -1:].expand_as(history_state)
                history_state = torch.where(
                    drop_mask[:, None, None],
                    current_state,
                    history_state,
                )
            if self.history_train_dropout_mode in {"zero_all", "video_only"}:
                repeated_current = current_frame.expand(
                    -1, -1, self.history_video_frames, -1, -1
                ).contiguous()
                history_video = torch.where(
                    drop_mask[:, None, None, None, None],
                    repeated_current,
                    history_video,
                )

        inputs["history_video"] = history_video
        inputs["history_action"] = history_action
        inputs["history_state"] = history_state
        for cache_key in ("history_dino_tokens", "video_dino_tokens"):
            cached = sample.get(cache_key, None)
            if cached is not None:
                if cached.ndim != 4 or cached.shape[0] != action.shape[0]:
                    raise ValueError(
                        f"Cached `{cache_key}` must be [B,T,N,D], got {tuple(cached.shape)}."
                    )
                inputs[cache_key] = cached.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                    non_blocking=True,
                )
        return inputs

    def _build_zero_action_t_mod(self, batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        zero_t = torch.zeros((batch_size,), dtype=dtype, device=device)
        t = self.action_expert.time_embedding(sinusoidal_embedding_1d(self.action_expert.freq_dim, zero_t))
        t_mod = self.action_expert.time_projection(t).unflatten(1, (6, self.action_expert.hidden_dim))
        return t_mod.unsqueeze(1).expand(-1, seq_len, -1, -1).contiguous()

    def _concat_history_action_pre(
        self,
        action_pre: dict[str, Any],
        history_prefix: torch.Tensor,
        *,
        include_semantic_query: bool = False,
    ) -> dict[str, Any]:
        prefix_len = int(history_prefix.shape[1])
        action_tokens = action_pre["tokens"]
        action_len = int(action_tokens.shape[1])
        extra_len = action_len if self.extra_prediction_action_tokens else 0
        semantic_query_len = (
            int(self.semantic_alignment_adapter.num_queries)
            if include_semantic_query and self.semantic_alignment_adapter is not None
            else 0
        )
        total_len = prefix_len + action_len + extra_len + semantic_query_len
        if total_len > self.action_expert.freqs.shape[0]:
            raise ValueError(
                f"History-prefixed action length {total_len} exceeds RoPE cache {self.action_expert.freqs.shape[0]}."
            )

        action_t_mod = action_pre["t_mod"]
        if action_t_mod.ndim == 3:
            action_t_mod = action_t_mod.unsqueeze(1).expand(-1, action_len, -1, -1).contiguous()
        elif action_t_mod.ndim != 4:
            raise ValueError(f"Unsupported action t_mod shape: {tuple(action_t_mod.shape)}")

        prefix_t_mod = self._build_zero_action_t_mod(
            batch_size=action_tokens.shape[0],
            seq_len=prefix_len,
            dtype=action_tokens.dtype,
            device=action_tokens.device,
        )
        context_mask = action_pre["context_mask"]
        if context_mask.ndim != 3:
            raise ValueError(f"`action_pre.context_mask` must be [B,S,L], got {tuple(context_mask.shape)}")
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
        semantic_query_start = extra_end
        semantic_query_end = semantic_query_start + semantic_query_len
        prediction_start = extra_start if self.prediction_action_group == "extra" else current_start
        prediction_end = extra_end if self.prediction_action_group == "extra" else current_end
        if semantic_query_len > 0:
            query_tokens = self.semantic_alignment_adapter.query_tokens(
                action_tokens.shape[0],
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            )
            token_chunks.append(query_tokens)
            t_mod_chunks.append(
                self._build_zero_action_t_mod(
                    batch_size=action_tokens.shape[0],
                    seq_len=semantic_query_len,
                    dtype=action_tokens.dtype,
                    device=action_tokens.device,
                )
            )
            context_mask_chunks.append(context_mask[:, :1, :].expand(-1, semantic_query_len, -1))

        merged = dict(action_pre)
        merged["tokens"] = torch.cat(token_chunks, dim=1)
        merged["freqs"] = self.action_expert.freqs[:total_len].view(total_len, 1, -1).to(action_tokens.device)
        merged["t_mod"] = torch.cat(t_mod_chunks, dim=1)
        merged["context_mask"] = torch.cat(context_mask_chunks, dim=1)
        merged["meta"] = dict(action_pre.get("meta", {}))
        merged["meta"]["history_prefix_len"] = prefix_len
        merged["meta"]["current_action_len"] = action_len
        merged["meta"]["extra_action_len"] = extra_len
        merged["meta"]["current_action_start"] = current_start
        merged["meta"]["current_action_end"] = current_end
        merged["meta"]["extra_action_start"] = extra_start
        merged["meta"]["extra_action_end"] = extra_end
        merged["meta"]["semantic_query_len"] = semantic_query_len
        merged["meta"]["semantic_query_start"] = semantic_query_start
        merged["meta"]["semantic_query_end"] = semantic_query_end
        merged["meta"]["prediction_action_group"] = self.prediction_action_group
        merged["meta"]["prediction_action_start"] = prediction_start
        merged["meta"]["prediction_action_end"] = prediction_end
        return merged

    def _build_history_attention_mask(
        self,
        *,
        video_seq_len: int,
        history_prefix_len: int,
        current_action_len: int,
        extra_action_len: int = 0,
        semantic_query_len: int = 0,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_action_len = history_prefix_len + current_action_len + extra_action_len + semantic_query_len
        total_seq_len = video_seq_len + total_action_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        prefix_start = video_seq_len
        prefix_end = prefix_start + history_prefix_len
        current_start = prefix_end
        current_end = current_start + current_action_len
        extra_start = current_end
        extra_end = extra_start + extra_action_len
        semantic_start = extra_end
        semantic_end = semantic_start + semantic_query_len

        if history_prefix_len > 0:
            mask[prefix_start:prefix_end, :first_frame_tokens] = True
            mask[prefix_start:prefix_end, prefix_start:prefix_end] = True
            if self.history_prefix_attend_current:
                mask[prefix_start:prefix_end, current_start:current_end] = True
        mask[current_start:current_end, :first_frame_tokens] = True
        mask[current_start:current_end, prefix_start:current_end] = True
        if extra_action_len > 0:
            mask[extra_start:extra_end, :first_frame_tokens] = True
            mask[extra_start:extra_end, extra_start:extra_end] = True
            if self.extra_prediction_attend_history:
                mask[extra_start:extra_end, prefix_start:prefix_end] = True
        if semantic_query_len > 0:
            mask[semantic_start:semantic_end, :first_frame_tokens] = True
            mask[semantic_start:semantic_end, prefix_start:semantic_end] = True
        return mask

    def _history_video_tokens(
        self,
        *,
        history_video: torch.Tensor,
        tiled: bool,
    ) -> torch.Tensor:
        history_latents = self._encode_video_latents(history_video, tiled=tiled)
        history_tokens = self.history_visual_tokenizer(history_latents)
        _patch_t, patch_h, patch_w = self.history_visual_tokenizer.patch_size
        _latent_t, latent_h, latent_w = history_latents.shape[2:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        first_frame_tokens = history_tokens[:, :tokens_per_frame, :]
        return first_frame_tokens

    @staticmethod
    def _history_tail_frame_offset(
        *,
        action_horizon: int,
        history_frame_count: int,
        selected_action_len: int,
        action_per_video_config: str | int | float,
    ) -> tuple[int, float]:
        if action_horizon <= 0 or history_frame_count <= 0 or selected_action_len <= 0:
            raise ValueError(
                "History-tail alignment lengths must be positive, got "
                f"horizon={action_horizon}, frames={history_frame_count}, selected={selected_action_len}."
            )
        if selected_action_len > action_horizon:
            raise ValueError(
                f"Selected history actions {selected_action_len} exceed horizon {action_horizon}."
            )
        if isinstance(action_per_video_config, str) and action_per_video_config.lower() == "auto":
            action_per_video = float(action_horizon) / float(history_frame_count)
        else:
            action_per_video = float(action_per_video_config)
        if not math.isfinite(action_per_video) or action_per_video <= 0.0:
            raise ValueError(f"Action/video ratio must be finite and positive, got {action_per_video}.")
        frame_offset = int(math.ceil(float(selected_action_len) / action_per_video))
        return min(max(frame_offset, 1), history_frame_count), action_per_video

    def _prepare_history_context(
        self,
        *,
        history_video: torch.Tensor,
        history_action: torch.Tensor,
        history_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict[str, int | float]]:
        adapter = self.history_adapter
        if history_state is not None:
            if history_state.ndim != 3 or history_state.shape[0] != history_action.shape[0]:
                raise ValueError(
                    "`history_state` must be [B,T+1,D], got "
                    f"{tuple(history_state.shape)} for history_action {tuple(history_action.shape)}."
                )
            if history_state.shape[1] != history_action.shape[1] + 1:
                raise ValueError(
                    "`history_state` must contain one more state than history actions, got "
                    f"{history_state.shape[1]} states and {history_action.shape[1]} actions."
                )
        history_start_state = None if history_state is None else history_state[:, 0]
        metrics: dict[str, int | float] = {
            "history_context_action_len": int(history_action.shape[1]),
            "history_context_video_start": 0,
            "history_context_video_len": int(history_video.shape[2]),
        }
        if not getattr(adapter, "action_random_tail_enabled", False):
            return history_video, history_action, history_start_state, metrics
        if not getattr(adapter, "action_random_tail_align_video_start", False):
            return history_video, history_action, history_start_state, metrics
        if not self.training and not getattr(adapter, "action_random_tail_eval", False):
            return history_video, history_action, history_start_state, metrics

        horizon = int(history_action.shape[1])
        valid_lengths = [
            int(length)
            for length in getattr(adapter, "action_random_tail_lengths", ())
            if 0 < int(length) <= horizon
        ]
        if not valid_lengths:
            return history_video, history_action, history_start_state, metrics
        action_len = int(random.choice(valid_lengths))
        action_start = horizon - action_len
        history_action = history_action[:, -action_len:]
        if history_state is not None:
            history_start_state = history_state[:, action_start]

        history_frame_count = max(int(history_video.shape[2]) - 1, 1)
        ratio_cfg = getattr(adapter, "action_random_tail_video_frame_ratio", "auto")
        frame_offset, action_per_video = self._history_tail_frame_offset(
            action_horizon=horizon,
            history_frame_count=history_frame_count,
            selected_action_len=action_len,
            action_per_video_config=ratio_cfg,
        )
        start_index = int(history_video.shape[2]) - 1 - frame_offset
        history_video = history_video[:, :, start_index:].contiguous()
        metrics = {
            "history_context_action_len": action_len,
            "history_context_video_start": start_index,
            "history_context_video_len": int(history_video.shape[2]),
            "history_context_action_per_video": float(action_per_video),
        }
        return history_video, history_action, history_start_state, metrics

    def _clean_action_estimate(
        self,
        *,
        noisy_action: torch.Tensor,
        pred_action: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (timestep_action / float(self.train_action_scheduler.num_train_timesteps)).to(
            device=noisy_action.device,
            dtype=noisy_action.dtype,
        )
        sigma = sigma.view(-1, *([1] * (noisy_action.ndim - 1)))
        return noisy_action - sigma * pred_action.to(device=noisy_action.device, dtype=noisy_action.dtype)

    def _compute_history_action_psd_loss(
        self,
        *,
        history_action: torch.Tensor,
        pred_action_clean: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.loss_lambda_action_psd <= 0:
            return pred_action_clean.new_zeros(())
        if history_action.ndim != 3 or pred_action_clean.ndim != 3 or target_action.ndim != 3:
            raise ValueError(
                "PSD action loss expects history, pred, and target action tensors shaped [B,T,D]."
            )
        if pred_action_clean.shape != target_action.shape:
            raise ValueError(
                f"`pred_action_clean` and `target_action` shape mismatch: "
                f"{tuple(pred_action_clean.shape)} vs {tuple(target_action.shape)}"
            )
        if history_action.shape[0] != target_action.shape[0] or history_action.shape[2] != target_action.shape[2]:
            raise ValueError(
                f"`history_action` must match batch/action dims of target action: "
                f"{tuple(history_action.shape)} vs {tuple(target_action.shape)}"
            )

        pred_current = pred_action_clean
        if action_is_pad is not None:
            if action_is_pad.shape != target_action.shape[:2]:
                raise ValueError(
                    f"`action_is_pad` must be [B,T], got {tuple(action_is_pad.shape)} "
                    f"for target {tuple(target_action.shape)}."
                )
            valid = (~action_is_pad).to(device=pred_current.device, dtype=pred_current.dtype).unsqueeze(-1)
            pred_current = pred_current * valid + target_action.to(dtype=pred_current.dtype) * (1.0 - valid)

        pred_seq = torch.cat([history_action.to(dtype=pred_current.dtype), pred_current], dim=1)
        target_seq = torch.cat(
            [
                history_action.to(dtype=target_action.dtype),
                target_action,
            ],
            dim=1,
        ).detach()
        if pred_seq.shape[1] < 2:
            return pred_action_clean.new_zeros(())

        losses = []
        for arm_slice in _action_arm_slices(
            pred_seq.shape[2],
            arm_dim=self.action_psd_arm_dim,
            mode=self.action_psd_arm_mode,
        ):
            losses.append(
                _psd_shape_loss(
                    pred_seq[:, :, arm_slice],
                    target_seq[:, :, arm_slice],
                    transform=self.action_psd_transform,
                    sample_period_frames=self.action_psd_sample_period_frames,
                    exclude_dc=self.action_psd_exclude_dc,
                    distance=self.action_psd_distance,
                    energy_mask=self.action_psd_energy_mask,
                    energy_mask_abs_floor=self.action_psd_energy_mask_abs_floor,
                    energy_mask_rel_floor=self.action_psd_energy_mask_rel_floor,
                    eps=self.action_psd_eps,
                )
            )
        return torch.stack(losses).mean()

    def training_loss(self, sample, tiled: bool = False):
        profile = (
            self.history_profile_steps > 0
            and int(getattr(self, "_train_step", 0)) < self.history_profile_steps
        )
        timings: dict[str, float] = {}
        shapes: dict[str, tuple[int, ...]] = {}

        def mark(name: str, start: float) -> float:
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            if profile:
                timings[name] = now - start
            return now

        t0 = time.perf_counter()
        inputs = self.build_inputs(sample, tiled=tiled)
        t = mark("build_inputs", t0)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        if profile:
            shapes["input_latents"] = tuple(input_latents.shape)
            shapes["history_video"] = tuple(inputs["history_video"].shape)
            shapes["action"] = tuple(action.shape)

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

        (
            context_history_video,
            context_history_action,
            context_history_start_state,
            history_context_metrics,
        ) = self._prepare_history_context(
            history_video=inputs["history_video"],
            history_action=inputs["history_action"],
            history_state=inputs["history_state"],
        )
        history_tokens = self._history_video_tokens(
            history_video=context_history_video,
            tiled=tiled,
        )
        if profile:
            shapes["history_tokens"] = tuple(history_tokens.shape)
        t = mark("history_video_tokens", t)
        history_prefix, _ = self.history_adapter(
            context_history_action,
            history_tokens,
            history_start_state=context_history_start_state,
        )
        if profile:
            shapes["history_prefix"] = tuple(history_prefix.shape)
        t = mark("history_adapter", t)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        if profile:
            shapes["video_tokens"] = tuple(video_pre["tokens"].shape)
        t = mark("video_pre", t)
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        t = mark("action_pre", t)
        action_pre_with_history = self._concat_history_action_pre(
            action_pre,
            history_prefix,
            include_semantic_query=(
                self.semantic_alignment_adapter is not None
                and self.loss_lambda_semantic_alignment > 0.0
            ),
        )

        video_tokens = video_pre["tokens"]
        action_meta = action_pre_with_history["meta"]
        attention_mask = self._build_history_attention_mask(
            video_seq_len=video_tokens.shape[1],
            history_prefix_len=history_prefix.shape[1],
            current_action_len=action.shape[1],
            extra_action_len=action.shape[1] if self.extra_prediction_action_tokens else 0,
            semantic_query_len=int(action_meta.get("semantic_query_len", 0)),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        t = mark("attention_mask", t)
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_pre_with_history["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre_with_history["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre_with_history["context"],
                    "mask": action_pre_with_history["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre_with_history["t_mod"],
            },
        )
        t = mark("mot", t)

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        loss_semantic_alignment = tokens_out["video"].new_zeros(())
        semantic_metrics: dict[str, torch.Tensor] = {}
        if self.semantic_alignment_adapter is not None and self.loss_lambda_semantic_alignment > 0.0:
            sem_start = int(action_meta["semantic_query_start"])
            sem_end = int(action_meta["semantic_query_end"])
            semantic_context = context
            semantic_context_mask = context_mask
            if self.semantic_alignment_target_context == "task":
                semantic_context = inputs.get("semantic_context", None)
                semantic_context_mask = inputs.get("semantic_context_mask", None)
                if semantic_context is None or semantic_context_mask is None:
                    if not self.semantic_alignment_fallback_to_prompt:
                        raise ValueError(
                            "`semantic_alignment.target_context=task` requires the batch to contain "
                            "`semantic_context` and `semantic_context_mask`."
                        )
                    semantic_context = context
                    semantic_context_mask = context_mask
            loss_semantic_alignment, semantic_metrics = self.semantic_alignment_adapter(
                query_hidden=tokens_out["action"][:, sem_start:sem_end],
                context=semantic_context,
                context_mask=semantic_context_mask,
            )

        def post_action_group(group: str) -> torch.Tensor:
            start = int(action_meta[f"{group}_action_start"])
            end = int(action_meta[f"{group}_action_end"])
            if end <= start:
                raise ValueError(f"Action group {group!r} is empty; check history dual-action config.")
            return self.action_expert.post_dit(tokens_out["action"][:, start:end], action_pre)

        pred_actions = {
            group: post_action_group(group)
            for group in self.supervise_action_groups
        }
        pred_action = pred_actions.get(self.prediction_action_group)
        if pred_action is None:
            pred_action = post_action_group(self.prediction_action_group)
        t = mark("post_dit", t)

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
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            target_action.device, dtype=target_action.dtype
        )
        action_group_losses: dict[str, torch.Tensor] = {}
        for group, pred_action_group in pred_actions.items():
            action_loss_token = F.mse_loss(
                pred_action_group.float(),
                target_action.float(),
                reduction="none",
            ).mean(dim=2)
            if action_is_pad is not None:
                valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)
            group_action_weight = action_weight.to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype,
            )
            action_group_losses[group] = (action_loss_per_sample * group_action_weight).mean()
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
        t = mark("loss", t)

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.loss_lambda_action_psd * loss_action_psd
            + self.loss_lambda_semantic_alignment * loss_semantic_alignment
        )
        if profile:
            timings["total_forward"] = time.perf_counter() - t0
            logger.info(
                "[profile:history_training_loss] step=%d timings=%s shapes=%s",
                int(getattr(self, "_train_step", 0)),
                {key: round(value, 4) for key, value in timings.items()},
                shapes,
            )
        loss_dict = {
            "loss_video": (self.loss_lambda_video * loss_video).detach(),
            "loss_action": (self.loss_lambda_action * loss_action).detach(),
            "loss_action_psd": (self.loss_lambda_action_psd * loss_action_psd).detach(),
            "loss_semantic_alignment": (
                self.loss_lambda_semantic_alignment * loss_semantic_alignment
            ).detach(),
            "history_fast_tokens": torch.as_tensor(
                float(self.history_fast_max_tokens), device=loss_total.device, dtype=loss_total.dtype
            ),
        }
        for metric_name, metric_value in semantic_metrics.items():
            loss_dict[metric_name] = metric_value.detach()
        for metric_name, metric_value in history_context_metrics.items():
            loss_dict[metric_name] = torch.as_tensor(
                float(metric_value),
                device=loss_total.device,
                dtype=loss_total.dtype,
            )
        for group, group_loss in action_group_losses.items():
            loss_dict[f"loss_action_{group}"] = (self.loss_lambda_action * group_loss).detach()
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_action_noise_with_history_cache(
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
        action_pre_with_history = self._concat_history_action_pre(action_pre, history_prefix)
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre_with_history["tokens"],
            action_freqs=action_pre_with_history["freqs"],
            action_t_mod=action_pre_with_history["t_mod"],
            action_context_payload={
                "context": action_pre_with_history["context"],
                "mask": action_pre_with_history["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        action_meta = action_pre_with_history["meta"]
        pred_start = int(action_meta["prediction_action_start"])
        pred_end = int(action_meta["prediction_action_end"])
        return self.action_expert.post_dit(action_tokens[:, pred_start:pred_end], action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_video: Optional[torch.Tensor] = None,
        history_action: Optional[torch.Tensor] = None,
        history_state: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'`.")

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
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
            history_action = torch.zeros((1, history_horizon, self.action_expert.action_dim), device=self.device, dtype=self.torch_dtype)
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

        history_video, history_action, history_start_state, _ = self._prepare_history_context(
            history_video=history_video,
            history_action=history_action,
            history_state=history_state,
        )
        history_tokens = self._history_video_tokens(
            history_video=history_video,
            tiled=tiled,
        )
        history_prefix, _ = self.history_adapter(
            history_action,
            history_tokens,
            history_start_state=history_start_state,
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
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_history_attention_mask(
            video_seq_len=video_seq_len,
            history_prefix_len=history_prefix.shape[1],
            current_action_len=latents_action.shape[1],
            extra_action_len=latents_action.shape[1] if self.extra_prediction_action_tokens else 0,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        guidance_scale = float(os.environ.get("WAM_HISTORY_GUIDANCE_SCALE", self.history_guidance_scale))
        if not math.isfinite(guidance_scale):
            raise ValueError(f"WAM history guidance scale must be finite, got {guidance_scale}.")
        use_history_guidance = guidance_scale != 1.0
        uncond_history_prefix = history_prefix[:, :0, :] if use_history_guidance else None
        uncond_attention_mask = None
        if use_history_guidance:
            uncond_attention_mask = self._build_history_attention_mask(
                video_seq_len=video_seq_len,
                history_prefix_len=0,
                current_action_len=latents_action.shape[1],
                extra_action_len=latents_action.shape[1] if self.extra_prediction_action_tokens else 0,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            if not use_history_guidance:
                pred_action = self._predict_action_noise_with_history_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    history_prefix=history_prefix,
                )
            elif guidance_scale == 0.0:
                pred_action = self._predict_action_noise_with_history_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=uncond_attention_mask,
                    video_seq_len=video_seq_len,
                    history_prefix=uncond_history_prefix,
                )
            else:
                pred_action_uncond = self._predict_action_noise_with_history_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=uncond_attention_mask,
                    video_seq_len=video_seq_len,
                    history_prefix=uncond_history_prefix,
                )
                pred_action_cond = self._predict_action_noise_with_history_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    history_prefix=history_prefix,
                )
                pred_action = pred_action_uncond + guidance_scale * (
                    pred_action_cond - pred_action_uncond
                )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "history_guidance_scale": guidance_scale,
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "history_visual_tokenizer": self.history_visual_tokenizer.state_dict(),
            "history_adapter": self.history_adapter.state_dict(),
            "history_config": self.history_config,
            "action_psd_config": self.action_psd_config,
            "semantic_alignment_config": self.semantic_alignment_config,
            "loss_lambda_action_psd": self.loss_lambda_action_psd,
            "loss_lambda_semantic_alignment": self.loss_lambda_semantic_alignment,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.semantic_alignment_adapter is not None:
            payload["semantic_alignment_adapter"] = self.semantic_alignment_adapter.state_dict()
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        modules = {
            "mot": self.mot,
            "history_visual_tokenizer": self.history_visual_tokenizer,
            "history_adapter": self.history_adapter,
        }
        for name in ("semantic_alignment_adapter", "proprio_encoder"):
            module = getattr(self, name, None)
            if module is not None:
                modules[name] = module
        missing_modules = sorted(set(modules) - set(payload))
        if missing_modules:
            raise RuntimeError(f"Checkpoint {path} is missing required modules: {missing_modules}")
        for name, module in modules.items():
            module.load_state_dict(payload[name], strict=True)
            logger.info("Strictly loaded %s from %s", name, path)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


def create_history_aware_wam(
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
    video_lora = _as_plain_dict(video_lora, "video_lora")

    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    return HistoryAwareWAM.from_wan22_pretrained(
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
        video_lora=video_lora,
        history_config=history,
        action_psd_config=action_psd,
        semantic_alignment_config=semantic_alignment,
    )


# Paper-facing aliases; legacy names remain available for checkpoint/config compatibility.
ActionExperienceEncoder = HistoryActionVideoAdapter
VisualHistoryCompressor = HistoryLatentVisualTokenizer
VisuallyConditionedActionBlock = HistoryActionVideoTransformerBlock
