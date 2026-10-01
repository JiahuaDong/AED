from __future__ import annotations

import builtins
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # Keep importable in lightweight environments.
    DictConfig = ()  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

from wam.utils.logging_config import get_logger

from .wam import _log_load_state_dict_result
from .visual_history import (
    HistoryActionVideoTransformerBlock,
    HistoryAwareWAM,
    _as_plain_dict,
)

logger = get_logger(__name__)


def _masked_mean(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    if mask.ndim != 2:
        raise ValueError(f"`mask` must be [B,L], got {tuple(mask.shape)}")
    valid = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def _temporal_token_slice(tokens: torch.Tensor, *, step: int, tokens_per_step: int) -> torch.Tensor:
    if tokens_per_step <= 0:
        raise ValueError(f"`tokens_per_step` must be positive, got {tokens_per_step}.")
    num_steps = tokens.shape[1] // tokens_per_step
    if tokens.shape[1] % tokens_per_step != 0 or num_steps <= 0:
        raise ValueError(
            f"`tokens` length {tokens.shape[1]} is not divisible by tokens_per_step={tokens_per_step}."
        )
    if step < 0:
        step = num_steps + step
    if not (0 <= step < num_steps):
        raise IndexError(f"Temporal token step {step} out of range for {num_steps} steps.")
    start = step * tokens_per_step
    return tokens[:, start : start + tokens_per_step]


class SETFrozenVisionStateTarget(nn.Module):
    """Frozen visual encoder target s* = LN(Pool(DINOv3(o))).

    The encoder and the optional dimensionality projection are intentionally
    non-trainable. This keeps SET targets fixed and prevents the transport loss
    from becoming small through a collapsed learned readout.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        model_id: str,
        source: str = "huggingface",
        local_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        image_size: int = 224,
        input_range: str = "minus_one_one",
        pool: str = "patch_mean",
        eps: float = 1e-6,
        projection_seed: int = 0,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.model_id = str(model_id)
        self.source = str(source).lower()
        self.local_path = None if local_path is None else str(local_path)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.image_size = int(image_size)
        self.input_range = str(input_range).lower()
        self.pool = str(pool).lower()
        self.projection_seed = int(projection_seed)
        if self.image_size <= 0:
            raise ValueError(f"`image_size` must be positive, got {image_size}.")
        if self.pool not in {"patch_mean", "mean", "cls", "pooler"}:
            raise ValueError(f"Unsupported SET frozen target pool={pool!r}.")
        if self.input_range not in {"minus_one_one", "zero_one"}:
            raise ValueError(f"Unsupported SET frozen target input_range={input_range!r}.")

        resolved_model = self._resolve_model_path()
        self.resolved_model = resolved_model

        try:
            import transformers
            from packaging.version import parse as parse_version
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise ImportError(
                "SET frozen DINO target requires transformers and packaging. "
                "Install a DINOv3-capable transformers release before training."
            ) from exc
        if parse_version(transformers.__version__) < parse_version("4.56.0"):
            raise ImportError(
                "SET frozen DINOv3 target requires transformers>=4.56.0. "
                f"Current transformers={transformers.__version__}. "
                "The older environment cannot instantiate DINOv3 AutoModel."
            )

        processor = AutoImageProcessor.from_pretrained(
            resolved_model,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.backbone = AutoModel.from_pretrained(
            resolved_model,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        ).eval()
        self.backbone.requires_grad_(False)

        backbone_dim = getattr(getattr(self.backbone, "config", None), "hidden_size", None)
        if backbone_dim is None:
            backbone_dim = getattr(getattr(self.backbone, "config", None), "embed_dim", None)
        if backbone_dim is None:
            raise ValueError(f"Could not infer hidden size from frozen vision model config: {resolved_model}")
        backbone_dim = int(backbone_dim)

        if backbone_dim == self.hidden_dim:
            self.projector = nn.Identity()
        else:
            projector = nn.Linear(backbone_dim, self.hidden_dim, bias=False)
            generator = torch.Generator(device="cpu").manual_seed(self.projection_seed)
            weight = torch.empty(self.hidden_dim, backbone_dim)
            nn.init.orthogonal_(weight, generator=generator)
            with torch.no_grad():
                projector.weight.copy_(weight)
            projector.requires_grad_(False)
            self.projector = projector
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=float(eps), elementwise_affine=False)

        mean = getattr(processor, "image_mean", None) or [0.485, 0.456, 0.406]
        std = getattr(processor, "image_std", None) or [0.229, 0.224, 0.225]
        self.register_buffer("image_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.to(dtype=torch_dtype)

    def train(self, mode: bool = True) -> "SETFrozenVisionStateTarget":
        # This module is a fixed teacher even while the enclosing WAM trains.
        super().train(False)
        self.backbone.eval()
        return self

    def _resolve_model_path(self) -> str:
        if self.local_path:
            path = Path(self.local_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"SET frozen target local_path does not exist: {path}")
            return str(path)
        if self.source in {"huggingface", "hf", "auto"}:
            return self.model_id
        if self.source in {"modelscope", "ms"}:
            try:
                from modelscope import snapshot_download
            except Exception as exc:  # pragma: no cover - optional dependency.
                raise ImportError("SET frozen target source='modelscope' requires the modelscope package.") from exc
            return snapshot_download(self.model_id, cache_dir=self.cache_dir)
        raise ValueError(f"Unsupported SET frozen target source={self.source!r}.")

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"SET frozen target expects frames [B,3,H,W], got {tuple(frames.shape)}")
        x = frames.detach().to(device=self.image_mean.device, dtype=torch.float32)
        if self.input_range == "minus_one_one":
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        x = (x - self.image_mean) / self.image_std.clamp_min(1.0e-6)
        param = next(self.backbone.parameters(), None)
        if param is not None:
            x = x.to(device=param.device, dtype=param.dtype)
        return x

    @torch.no_grad()
    def patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        pixel_values = self._preprocess(frames)
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        hidden = outputs.last_hidden_state
        config = getattr(self.backbone, "config", None)
        num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        prefix_tokens = 1 + max(0, num_register_tokens)
        if hidden.shape[1] <= prefix_tokens:
            prefix_tokens = 1 if hidden.shape[1] > 1 else 0
        tokens = hidden[:, prefix_tokens:]
        if tokens.shape[1] <= 0:
            raise ValueError(
                "Frozen vision target produced no patch tokens after removing "
                f"{prefix_tokens} prefix token(s); hidden shape={tuple(hidden.shape)}."
            )
        projector_param = next(self.projector.parameters(), None)
        projector_dtype = projector_param.dtype if projector_param is not None else tokens.dtype
        tokens = self.projector(tokens.to(dtype=projector_dtype))
        return self.output_norm(tokens).detach()

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        pixel_values = self._preprocess(frames)
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        if self.pool == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state
            if self.pool == "cls":
                pooled = hidden[:, 0]
            elif self.pool == "patch_mean" and hidden.shape[1] > 1:
                pooled = hidden[:, 1:].mean(dim=1)
            else:
                pooled = hidden.mean(dim=1)
        pooled = self.projector(pooled.to(dtype=next(self.projector.parameters(), pooled).dtype))
        return self.output_norm(pooled).detach()


class SETFrozenLingBotVisionTarget(nn.Module):
    """Frozen LingBot-Vision patch target with the SET target interface."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        model_id: str = "robbyant/lingbot-vision-vit-base",
        variant: str = "base",
        source: str = "huggingface",
        local_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        image_size: int = 224,
        input_range: str = "minus_one_one",
        pool: str = "patch_mean",
        eps: float = 1e-6,
        projection_seed: int = 0,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.model_id = str(model_id)
        self.variant = str(variant).lower()
        self.source = str(source).lower()
        self.local_path = None if local_path is None else str(local_path)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.image_size = int(image_size)
        self.input_range = str(input_range).lower()
        self.pool = str(pool).lower()
        self.projection_seed = int(projection_seed)
        if self.image_size <= 0:
            raise ValueError(f"`image_size` must be positive, got {image_size}.")
        if self.pool not in {"patch_mean", "mean", "cls", "pooler"}:
            raise ValueError(f"Unsupported LingBot-Vision target pool={pool!r}.")
        if self.input_range not in {"minus_one_one", "zero_one"}:
            raise ValueError(f"Unsupported LingBot-Vision target input_range={input_range!r}.")

        resolved_model = self._resolve_model_path()
        self.resolved_model = resolved_model
        try:
            from lingbot_vision import load_pretrained_backbone
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise ImportError(
                "LingBot-Vision SET targets require the official lingbot-vision package. "
                "Install WAM/third_party/lingbot-vision with --no-deps."
            ) from exc

        self.backbone, backbone_dim = load_pretrained_backbone(
            repo_id_or_path=resolved_model,
            variant=self.variant,
            device="cpu",
            dtype=torch.float32,
            cache_dir=self.cache_dir,
            local_files_only=bool(self.local_path),
            verbose=True,
        )
        self.backbone.eval().requires_grad_(False)
        backbone_dim = int(backbone_dim)

        if backbone_dim == self.hidden_dim:
            self.projector = nn.Identity()
        else:
            projector = nn.Linear(backbone_dim, self.hidden_dim, bias=False)
            generator = torch.Generator(device="cpu").manual_seed(self.projection_seed)
            weight = torch.empty(self.hidden_dim, backbone_dim)
            nn.init.orthogonal_(weight, generator=generator)
            with torch.no_grad():
                projector.weight.copy_(weight)
            projector.requires_grad_(False)
            self.projector = projector
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=float(eps), elementwise_affine=False)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.to(dtype=torch_dtype)

    def train(self, mode: bool = True) -> "SETFrozenLingBotVisionTarget":
        # Keep the frozen target deterministic when parent model.train() recurses.
        super().train(False)
        self.backbone.eval()
        return self

    def _resolve_model_path(self) -> str:
        if self.local_path:
            path = Path(self.local_path).expanduser()
            if not path.is_dir() or not (path / "model.pt").is_file():
                raise FileNotFoundError(
                    "LingBot-Vision local_path must contain model.pt: "
                    f"{path}"
                )
            return str(path)
        if self.source in {"huggingface", "hf", "auto"}:
            return self.model_id
        if self.source in {"modelscope", "ms"}:
            try:
                from modelscope import snapshot_download
            except Exception as exc:  # pragma: no cover - optional dependency.
                raise ImportError("LingBot-Vision source='modelscope' requires modelscope.") from exc
            return snapshot_download(self.model_id, cache_dir=self.cache_dir)
        raise ValueError(f"Unsupported LingBot-Vision target source={self.source!r}.")

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"LingBot-Vision target expects frames [B,3,H,W], got {tuple(frames.shape)}")
        x = frames.detach().to(device=self.image_mean.device, dtype=torch.float32)
        if self.input_range == "minus_one_one":
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        x = (x - self.image_mean) / self.image_std.clamp_min(1.0e-6)
        param = next(self.backbone.parameters(), None)
        if param is not None:
            x = x.to(device=param.device, dtype=param.dtype)
        return x

    @torch.no_grad()
    def _features(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(self._preprocess(frames), is_training=True)

    @torch.no_grad()
    def patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        outputs = self._features(frames)
        tokens = outputs["x_norm_patchtokens"]
        projector_param = next(self.projector.parameters(), None)
        projector_dtype = projector_param.dtype if projector_param is not None else tokens.dtype
        tokens = self.projector(tokens.to(dtype=projector_dtype))
        return self.output_norm(tokens).detach()

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        outputs = self._features(frames)
        if self.pool in {"cls", "pooler"}:
            pooled = outputs["x_norm_clstoken"]
        else:
            pooled = outputs["x_norm_patchtokens"].mean(dim=1)
        projector_param = next(self.projector.parameters(), None)
        projector_dtype = projector_param.dtype if projector_param is not None else pooled.dtype
        pooled = self.projector(pooled.to(dtype=projector_dtype))
        return self.output_norm(pooled).detach()


class SETSemanticStateReadout(nn.Module):
    """Instruction-conditioned readout R_psi(z, c) -> semantic state s."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        text_dim: int,
        attn_head_dim: int,
        num_heads: int,
        num_queries: int = 4,
        eps: float = 1e-6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.text_dim = int(text_dim)
        self.num_queries = int(num_queries)
        if self.num_queries <= 0:
            raise ValueError(f"`num_queries` must be positive, got {num_queries}.")
        self.text_norm = nn.LayerNorm(self.text_dim, eps=float(eps))
        self.text_to_hidden = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
        )
        self.queries = nn.Parameter(torch.randn(1, self.num_queries, self.hidden_dim) * 0.02)
        self.block = HistoryActionVideoTransformerBlock(
            hidden_dim=self.hidden_dim,
            attn_head_dim=attn_head_dim,
            num_heads=num_heads,
            eps=float(eps),
            ffn_mlp_ratio=float(mlp_ratio),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
        )

    def text_condition(self, context: torch.Tensor, context_mask: Optional[torch.Tensor]) -> torch.Tensor:
        text = self.text_norm(context.to(dtype=self.text_norm.weight.dtype))
        return self.text_to_hidden(_masked_mean(text, context_mask))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visual_tokens.ndim != 3:
            raise ValueError(f"`visual_tokens` must be [B,N,D], got {tuple(visual_tokens.shape)}")
        text_hidden = self.text_condition(context, context_mask).to(
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        query = self.queries.to(device=visual_tokens.device, dtype=visual_tokens.dtype).expand(
            visual_tokens.shape[0], -1, -1
        )
        query = query + text_hidden.unsqueeze(1)
        state_tokens = self.block(query, visual_tokens)
        state = self.out(state_tokens.mean(dim=1))
        return state, text_hidden


class SETEventEncoder(nn.Module):
    """Encode an observation-action segment into event coordinates alpha and prefix tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_event_queries: int,
        attn_head_dim: int,
        num_heads: int,
        num_layers: int = 1,
        eps: float = 1e-6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_event_queries = int(num_event_queries)
        if self.num_event_queries <= 0:
            raise ValueError(f"`num_event_queries` must be positive, got {num_event_queries}.")
        self.event_queries = nn.Parameter(torch.randn(1, self.num_event_queries, self.hidden_dim) * 0.02)
        self.state_to_token = nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
        )
        self.text_to_query = nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                HistoryActionVideoTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    eps=float(eps),
                    ffn_mlp_ratio=float(mlp_ratio),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, 1),
        )
        self.prefix_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=float(eps)),
        )

    def forward(
        self,
        *,
        start_state: torch.Tensor,
        action_tokens: torch.Tensor,
        text_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if start_state.ndim != 2:
            raise ValueError(f"`start_state` must be [B,D], got {tuple(start_state.shape)}")
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,N,D], got {tuple(action_tokens.shape)}")
        if text_hidden.ndim != 2:
            raise ValueError(f"`text_hidden` must be [B,D], got {tuple(text_hidden.shape)}")
        dtype = action_tokens.dtype
        device = action_tokens.device
        query = self.event_queries.to(device=device, dtype=dtype).expand(action_tokens.shape[0], -1, -1)
        query = query + self.text_to_query(text_hidden.to(device=device, dtype=dtype)).unsqueeze(1)
        state_token = self.state_to_token(start_state.to(device=device, dtype=dtype)).unsqueeze(1)
        text_token = text_hidden.to(device=device, dtype=dtype).unsqueeze(1)
        memory = torch.cat([state_token, text_token, action_tokens], dim=1)
        event_tokens = query
        for block in self.blocks:
            event_tokens = block(event_tokens, memory)
        alpha = self.alpha_head(event_tokens).squeeze(-1)
        prefix = self.prefix_projector(event_tokens)
        return prefix, alpha, event_tokens


class SETTransport(nn.Module):
    """State-conditioned event transport T_phi(s, alpha, c)."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_event_queries: int,
        eps: float = 1e-6,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_event_queries = int(num_event_queries)
        hidden = max(self.hidden_dim, int(round(self.hidden_dim * float(mlp_ratio))))
        self.input_norm = nn.LayerNorm(self.hidden_dim * 2, eps=float(eps))
        self.vector_field = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, self.num_event_queries * self.hidden_dim),
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim, eps=float(eps))

    def forward(self, state: torch.Tensor, alpha: torch.Tensor, text_hidden: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or text_hidden.ndim != 2 or alpha.ndim != 2:
            raise ValueError(
                "SET transport expects state/text [B,D] and alpha [B,M], got "
                f"state={tuple(state.shape)} text={tuple(text_hidden.shape)} alpha={tuple(alpha.shape)}"
            )
        x = self.input_norm(torch.cat([state, text_hidden.to(dtype=state.dtype)], dim=-1))
        fields = self.vector_field(x).view(state.shape[0], self.num_event_queries, self.hidden_dim)
        delta = (alpha.to(dtype=state.dtype).unsqueeze(-1) * fields).sum(dim=1)
        return self.out_norm(state + delta)


class SETWAM(HistoryAwareWAM):
    """History-aware WAM with Semantic Event Transport prefixes.

    The same `_build_set_history_event` function is used by training and
    inference. Current event query tokens are also present in both modes, so
    action denoising sees a consistent action-branch sequence layout.
    """

    def __init__(
        self,
        *args,
        set_config: Optional[dict[str, Any]] = None,
        loss_lambda_set: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.configure_set(set_config=set_config, loss_lambda_set=loss_lambda_set)

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        set_config: Optional[dict[str, Any]] = None,
        loss_lambda_set: float = 0.0,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(*args, **kwargs)
        model.configure_set(set_config=set_config, loss_lambda_set=loss_lambda_set)
        return model

    def configure_set(self, *, set_config: Optional[dict[str, Any]], loss_lambda_set: float) -> None:
        cfg = dict(set_config or {})
        self.set_config = cfg
        self.set_enabled = bool(cfg.get("enabled", True))
        self.loss_lambda_set = float(loss_lambda_set)
        self.set_num_state_queries = int(cfg.get("num_state_queries", 4))
        self.set_num_event_queries = int(cfg.get("num_event_queries", cfg.get("num_events", 8)))
        self.set_loss_history_weight = float(cfg.get("history_loss_weight", 0.25))
        self.set_loss_direct_weight = float(cfg.get("direct_loss_weight", 1.0))
        self.set_loss_sequence_weight = float(cfg.get("sequence_loss_weight", 1.0))
        self.set_alpha_loss_weight = float(cfg.get("alpha_loss_weight", 0.0))
        self.set_bidirectional_event_query = bool(cfg.get("bidirectional_event_query", True))
        self.set_inject_history_prefix = bool(
            cfg.get("inject_history_prefix", cfg.get("inject_prefix_into_action", True))
        )
        self.set_history_bias_enabled = bool(cfg.get("history_bias_enabled", False))
        self.set_history_bias_gate_init = float(cfg.get("history_bias_gate_init", 0.0))
        hidden_dim = int(self.action_expert.hidden_dim)
        eps = float(cfg.get("eps", getattr(self.action_expert.blocks[0], "norm1").eps))
        target_cfg = dict(cfg.get("target", {}))
        self.set_target_type = str(target_cfg.get("type", "readout")).lower()
        self.set_state_readout = SETSemanticStateReadout(
            hidden_dim=hidden_dim,
            text_dim=int(self.text_dim),
            attn_head_dim=int(self.action_expert.attn_head_dim),
            num_heads=int(self.action_expert.num_heads),
            num_queries=self.set_num_state_queries,
            eps=eps,
            mlp_ratio=float(cfg.get("state_mlp_ratio", 4.0)),
        )
        self.set_event_encoder = SETEventEncoder(
            hidden_dim=hidden_dim,
            num_event_queries=self.set_num_event_queries,
            attn_head_dim=int(self.action_expert.attn_head_dim),
            num_heads=int(self.action_expert.num_heads),
            num_layers=int(cfg.get("event_layers", 1)),
            eps=eps,
            mlp_ratio=float(cfg.get("event_mlp_ratio", 4.0)),
        )
        self.set_current_event_query = nn.Parameter(
            torch.randn(1, self.set_num_event_queries, hidden_dim, dtype=torch.float32) * 0.02
        )
        self.set_query_to_alpha = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=eps),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 1),
        )
        self.set_transport = SETTransport(
            hidden_dim=hidden_dim,
            num_event_queries=self.set_num_event_queries,
            eps=eps,
            mlp_ratio=float(cfg.get("transport_mlp_ratio", 2.0)),
        )
        if self.set_history_bias_enabled:
            bias_hidden = max(hidden_dim, int(round(hidden_dim * float(cfg.get("history_bias_mlp_ratio", 1.0)))))
            self.set_history_bias_projector = nn.Sequential(
                nn.LayerNorm(hidden_dim, eps=eps),
                nn.Linear(hidden_dim, bias_hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(bias_hidden, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=eps),
            )
            self.set_history_bias_gate = nn.Parameter(
                torch.tensor(self.set_history_bias_gate_init, dtype=torch.float32)
            )
        else:
            self.set_history_bias_projector = None
        if self.set_target_type in {"frozen_dino", "dino", "frozen_vision"}:
            self.set_target_encoder = SETFrozenVisionStateTarget(
                hidden_dim=hidden_dim,
                model_id=str(target_cfg.get("model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m")),
                source=str(target_cfg.get("source", "huggingface")),
                local_path=target_cfg.get("local_path", None),
                cache_dir=target_cfg.get("cache_dir", None),
                image_size=int(target_cfg.get("image_size", 224)),
                input_range=str(target_cfg.get("input_range", "minus_one_one")),
                pool=str(target_cfg.get("pool", "patch_mean")),
                eps=eps,
                projection_seed=int(target_cfg.get("projection_seed", 0)),
                torch_dtype=self.torch_dtype,
            )
        elif self.set_target_type in {"readout", "learned_readout", "learned"}:
            self.set_target_encoder = None
        else:
            raise ValueError(
                "Unsupported SET target.type=%r. Expected one of readout, frozen_dino, frozen_vision."
                % self.set_target_type
            )
        self.set_state_readout.to(device=self.device, dtype=self.torch_dtype)
        self.set_event_encoder.to(device=self.device, dtype=self.torch_dtype)
        self.set_transport.to(device=self.device, dtype=self.torch_dtype)
        self.set_query_to_alpha.to(device=self.device, dtype=self.torch_dtype)
        if self.set_history_bias_projector is not None:
            self.set_history_bias_projector.to(device=self.device, dtype=self.torch_dtype)
            self.set_history_bias_gate.data = self.set_history_bias_gate.data.to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        if self.set_target_encoder is not None:
            self.set_target_encoder.to(device=self.device, dtype=self.torch_dtype)
            self.set_target_encoder.eval().requires_grad_(False)
        self.set_current_event_query.data = self.set_current_event_query.data.to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        logger.info(
            "SET-WAM config: enabled=%s lambda_set=%.4g events=%d state_queries=%d "
            "target=%s loss_weights(history=%.3g,direct=%.3g,sequence=%.3g,alpha=%.3g) "
            "bidirectional_query=%s inject_history_prefix=%s history_bias=%s",
            self.set_enabled,
            self.loss_lambda_set,
            self.set_num_event_queries,
            self.set_num_state_queries,
            self.set_target_type,
            self.set_loss_history_weight,
            self.set_loss_direct_weight,
            self.set_loss_sequence_weight,
            self.set_alpha_loss_weight,
            self.set_bidirectional_event_query,
            self.set_inject_history_prefix,
            self.set_history_bias_enabled,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for module_name in (
            "set_state_readout",
            "set_event_encoder",
            "set_transport",
            "set_query_to_alpha",
            "set_history_bias_projector",
        ):
            module = getattr(self, module_name, None)
            if module is not None:
                module.to(*args, **kwargs)
        if hasattr(self, "set_history_bias_gate"):
            self.set_history_bias_gate.data = self.set_history_bias_gate.data.to(*args, **kwargs)
        if getattr(self, "set_target_encoder", None) is not None:
            self.set_target_encoder.to(*args, **kwargs)
            self.set_target_encoder.eval().requires_grad_(False)
        if hasattr(self, "set_current_event_query"):
            self.set_current_event_query.data = self.set_current_event_query.data.to(*args, **kwargs)
        return self

    def _set_uses_frozen_target(self) -> bool:
        return getattr(self, "set_target_encoder", None) is not None

    def extra_trainable_modules(self):
        modules = list(super().extra_trainable_modules())
        if self._set_uses_frozen_target():
            modules.extend([self.set_state_readout.text_norm, self.set_state_readout.text_to_hidden])
        else:
            modules.append(self.set_state_readout)
        modules.extend([self.set_event_encoder, self.set_transport, self.set_query_to_alpha])
        if getattr(self, "set_history_bias_projector", None) is not None:
            modules.append(self.set_history_bias_projector)
        return modules

    def extra_trainable_parameters(self):
        params = list(super().extra_trainable_parameters())
        if self._set_uses_frozen_target():
            params.extend(self.set_state_readout.text_norm.parameters())
            params.extend(self.set_state_readout.text_to_hidden.parameters())
        else:
            params.extend(self.set_state_readout.parameters())
        params.extend(self.set_event_encoder.parameters())
        params.extend(self.set_transport.parameters())
        params.extend(self.set_query_to_alpha.parameters())
        params.append(self.set_current_event_query)
        if getattr(self, "set_history_bias_projector", None) is not None:
            params.extend(self.set_history_bias_projector.parameters())
            params.append(self.set_history_bias_gate)
        return params

    def _visual_tokens_from_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        tokens = self.history_visual_tokenizer(latents)
        _patch_t, patch_h, patch_w = self.history_visual_tokenizer.patch_size
        _latent_t, latent_h, latent_w = latents.shape[2:]
        tokens_per_step = (latent_h // patch_h) * (latent_w // patch_w)
        return tokens, tokens_per_step

    def _set_text_hidden(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        text_hidden = self.set_state_readout.text_condition(context, context_mask)
        return text_hidden.to(device=device, dtype=dtype)

    def _frozen_video_state(self, video: torch.Tensor, *, step: int) -> torch.Tensor:
        if not self._set_uses_frozen_target():
            raise RuntimeError("_frozen_video_state called but SET target is not frozen.")
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"SET frozen target expects video [B,3,T,H,W], got {tuple(video.shape)}")
        if step < 0:
            step = int(video.shape[2]) + step
        if not (0 <= step < int(video.shape[2])):
            raise IndexError(f"SET frozen target frame step {step} out of range for video T={video.shape[2]}.")
        frame = video[:, :, step].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        state = self.set_target_encoder(frame)
        return state.to(device=self.device, dtype=self.torch_dtype)

    def _encode_event_actions(self, actions: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        token_ids, token_mask = self.history_adapter.fast_tokenizer.encode(actions)
        token_ids = token_ids.to(device=device)
        token_mask = token_mask.to(device=device)
        tokens = self.history_adapter.action_token_embedding(token_ids)
        tokens = tokens * token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        tokens = self.history_adapter.action_token_projector(tokens)
        return tokens.to(dtype=dtype) * token_mask.unsqueeze(-1).to(dtype=dtype)

    def _build_set_history_event(
        self,
        *,
        history_action: torch.Tensor,
        history_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        tiled: bool,
    ) -> dict[str, torch.Tensor]:
        if self._set_uses_frozen_target():
            state_start = self._frozen_video_state(history_video, step=0)
            state_end = self._frozen_video_state(history_video, step=-1)
            text_hidden = self._set_text_hidden(
                context,
                context_mask,
                device=state_start.device,
                dtype=state_start.dtype,
            )
            tokens_per_step = 0
            action_device = state_start.device
            action_dtype = state_start.dtype
        else:
            history_latents = self._encode_video_latents(history_video, tiled=tiled)
            history_tokens, tokens_per_step = self._visual_tokens_from_latents(history_latents)
            start_tokens = _temporal_token_slice(history_tokens, step=0, tokens_per_step=tokens_per_step)
            end_tokens = _temporal_token_slice(history_tokens, step=-1, tokens_per_step=tokens_per_step)
            state_start, text_hidden = self.set_state_readout(start_tokens, context=context, context_mask=context_mask)
            state_end, _ = self.set_state_readout(end_tokens, context=context, context_mask=context_mask)
            action_device = history_tokens.device
            action_dtype = history_tokens.dtype
        action_tokens = self._encode_event_actions(
            history_action,
            device=action_device,
            dtype=action_dtype,
        )
        prefix, alpha, event_tokens = self.set_event_encoder(
            start_state=state_start,
            action_tokens=action_tokens,
            text_hidden=text_hidden,
        )
        pred_end = self.set_transport(state_start, alpha, text_hidden)
        return {
            "prefix": prefix,
            "alpha": alpha,
            "event_tokens": event_tokens,
            "state_start": state_start,
            "state_end": state_end,
            "text_hidden": text_hidden,
            "pred_end": pred_end,
            "tokens_per_step": torch.as_tensor(tokens_per_step, device=prefix.device),
        }

    def _set_action_prefix_for_mot(self, event_prefix: torch.Tensor) -> torch.Tensor:
        if self.set_inject_history_prefix:
            return event_prefix
        return event_prefix[:, :0, :]

    def _set_history_bias_gate_value(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.set_history_bias_enabled:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.tanh(self.set_history_bias_gate.to(device=device, dtype=dtype))

    def _apply_set_history_bias(
        self,
        action_tokens: torch.Tensor,
        history_event_prefix: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.set_history_bias_enabled:
            return action_tokens
        if history_event_prefix is None or history_event_prefix.shape[1] == 0:
            return action_tokens
        source = history_event_prefix.to(device=action_tokens.device, dtype=action_tokens.dtype).mean(dim=1)
        bias = self.set_history_bias_projector(source).unsqueeze(1)
        gate = self._set_history_bias_gate_value(device=action_tokens.device, dtype=action_tokens.dtype)
        return action_tokens + gate.view(1, 1, 1) * bias

    def _concat_set_action_pre(
        self,
        action_pre: dict[str, Any],
        event_prefix: torch.Tensor,
        *,
        include_event_query: bool = True,
        history_event_prefix_for_bias: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        prefix_len = int(event_prefix.shape[1])
        action_tokens = self._apply_set_history_bias(
            action_pre["tokens"],
            history_event_prefix_for_bias if history_event_prefix_for_bias is not None else event_prefix,
        )
        action_len = int(action_tokens.shape[1])
        extra_len = action_len if self.extra_prediction_action_tokens else 0
        event_query_len = self.set_num_event_queries if include_event_query else 0
        total_len = prefix_len + action_len + extra_len + event_query_len
        if total_len > self.action_expert.freqs.shape[0]:
            raise ValueError(f"SET action length {total_len} exceeds RoPE cache {self.action_expert.freqs.shape[0]}.")

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

        token_chunks = [event_prefix, action_tokens]
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
        event_query_start = extra_end
        event_query_end = event_query_start + event_query_len
        prediction_start = extra_start if self.prediction_action_group == "extra" else current_start
        prediction_end = extra_end if self.prediction_action_group == "extra" else current_end

        if event_query_len > 0:
            query_tokens = self.set_current_event_query.to(
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            ).expand(action_tokens.shape[0], -1, -1)
            token_chunks.append(query_tokens)
            t_mod_chunks.append(
                self._build_zero_action_t_mod(
                    batch_size=action_tokens.shape[0],
                    seq_len=event_query_len,
                    dtype=action_tokens.dtype,
                    device=action_tokens.device,
                )
            )
            context_mask_chunks.append(context_mask[:, :1, :].expand(-1, event_query_len, -1))

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
                "set_event_query_len": event_query_len,
                "set_event_query_start": event_query_start,
                "set_event_query_end": event_query_end,
                "prediction_action_group": self.prediction_action_group,
                "prediction_action_start": prediction_start,
                "prediction_action_end": prediction_end,
                "set_history_bias_enabled": self.set_history_bias_enabled,
            }
        )
        return merged

    def _build_set_attention_mask(
        self,
        *,
        video_seq_len: int,
        event_prefix_len: int,
        current_action_len: int,
        extra_action_len: int,
        event_query_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = self._build_history_attention_mask(
            video_seq_len=video_seq_len,
            history_prefix_len=event_prefix_len,
            current_action_len=current_action_len,
            extra_action_len=extra_action_len,
            semantic_query_len=event_query_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        if self.set_bidirectional_event_query and event_query_len > 0:
            prefix_start = video_seq_len
            current_start = prefix_start + event_prefix_len
            current_end = current_start + current_action_len
            extra_start = current_end
            extra_end = extra_start + extra_action_len
            query_start = extra_end
            query_end = query_start + event_query_len
            mask[current_start:current_end, query_start:query_end] = True
            if extra_action_len > 0:
                mask[extra_start:extra_end, query_start:query_end] = True
        return mask

    def _event_query_alpha(self, query_hidden: torch.Tensor) -> torch.Tensor:
        if query_hidden.ndim != 3:
            raise ValueError(f"`query_hidden` must be [B,M,D], got {tuple(query_hidden.shape)}")
        return self.set_query_to_alpha(query_hidden).squeeze(-1)

    def _compute_set_loss(
        self,
        *,
        input_latents: torch.Tensor,
        video: torch.Tensor,
        history_event: dict[str, torch.Tensor],
        current_alpha: torch.Tensor,
        history_action: torch.Tensor,
        current_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.loss_lambda_set <= 0.0:
            zero = input_latents.new_zeros(())
            return zero, {}
        if self._set_uses_frozen_target():
            state_current = self._frozen_video_state(video, step=0)
            state_future = self._frozen_video_state(video, step=-1)
            text_hidden = history_event["text_hidden"]
        else:
            video_tokens, tokens_per_step = self._visual_tokens_from_latents(input_latents)
            current_tokens = _temporal_token_slice(video_tokens, step=0, tokens_per_step=tokens_per_step)
            future_tokens = _temporal_token_slice(video_tokens, step=-1, tokens_per_step=tokens_per_step)
            state_current, text_hidden = self.set_state_readout(current_tokens, context=context, context_mask=context_mask)
            state_future, _ = self.set_state_readout(future_tokens, context=context, context_mask=context_mask)

        full_action = torch.cat([history_action.to(dtype=current_action.dtype), current_action], dim=1)
        full_action_tokens = self._encode_event_actions(
            full_action,
            device=input_latents.device,
            dtype=input_latents.dtype,
        )
        _, direct_alpha, _ = self.set_event_encoder(
            start_state=history_event["state_start"],
            action_tokens=full_action_tokens,
            text_hidden=history_event["text_hidden"],
        )

        hist_pred = history_event["pred_end"]
        seq_pred = self.set_transport(hist_pred, current_alpha, history_event["text_hidden"])
        direct_pred = self.set_transport(history_event["state_start"], direct_alpha, history_event["text_hidden"])
        target_future_state = state_future.detach()
        target_current_state = state_current.detach()

        loss_seq = F.mse_loss(seq_pred.float(), target_future_state.float(), reduction="mean")
        loss_direct = F.mse_loss(direct_pred.float(), target_future_state.float(), reduction="mean")
        loss_history = F.mse_loss(hist_pred.float(), target_current_state.float(), reduction="mean")
        loss_alpha = F.mse_loss(current_alpha.float(), direct_alpha.detach().float(), reduction="mean")
        loss = (
            self.set_loss_sequence_weight * loss_seq
            + self.set_loss_direct_weight * loss_direct
            + self.set_loss_history_weight * loss_history
            + self.set_alpha_loss_weight * loss_alpha
        )
        with torch.no_grad():
            metrics = {
                "loss_set_seq_raw": loss_seq.detach(),
                "loss_set_direct_raw": loss_direct.detach(),
                "loss_set_history_raw": loss_history.detach(),
                "loss_set_alpha_raw": loss_alpha.detach(),
                "set_alpha_abs_mean": current_alpha.detach().abs().mean(),
                "set_direct_alpha_abs_mean": direct_alpha.detach().abs().mean(),
                "set_target_future_norm": target_future_state.detach().float().norm(dim=-1).mean(),
                "set_target_current_norm": target_current_state.detach().float().norm(dim=-1).mean(),
                "set_seq_state_cos": (
                    F.normalize(seq_pred.float(), dim=-1) * F.normalize(target_future_state.float(), dim=-1)
                ).sum(dim=-1).mean(),
            }
        return loss, metrics

    def training_loss(self, sample, tiled: bool = False):
        if not self.set_enabled:
            return super().training_loss(sample, tiled=tiled)

        t0 = time.perf_counter()
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
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

        history_event = self._build_set_history_event(
            history_action=inputs["history_action"],
            history_video=inputs["history_video"],
            context=context,
            context_mask=context_mask,
            tiled=tiled,
        )
        full_event_prefix = history_event["prefix"]
        event_prefix = self._set_action_prefix_for_mot(full_event_prefix)

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
        action_pre_with_event = self._concat_set_action_pre(
            action_pre,
            event_prefix,
            include_event_query=True,
            history_event_prefix_for_bias=full_event_prefix,
        )
        video_tokens = video_pre["tokens"]
        action_meta = action_pre_with_event["meta"]
        attention_mask = self._build_set_attention_mask(
            video_seq_len=video_tokens.shape[1],
            event_prefix_len=event_prefix.shape[1],
            current_action_len=action.shape[1],
            extra_action_len=action.shape[1] if self.extra_prediction_action_tokens else 0,
            event_query_len=int(action_meta.get("set_event_query_len", 0)),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_pre_with_event["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre_with_event["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {
                    "context": action_pre_with_event["context"],
                    "mask": action_pre_with_event["context_mask"],
                },
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre_with_event["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        event_start = int(action_meta["set_event_query_start"])
        event_end = int(action_meta["set_event_query_end"])
        current_alpha = self._event_query_alpha(tokens_out["action"][:, event_start:event_end])
        loss_set_raw, set_metrics = self._compute_set_loss(
            input_latents=input_latents,
            video=sample["video"],
            history_event=history_event,
            current_alpha=current_alpha,
            history_action=inputs["history_action"],
            current_action=action,
            context=context,
            context_mask=context_mask,
        )

        def post_action_group(group: str) -> torch.Tensor:
            start = int(action_meta[f"{group}_action_start"])
            end = int(action_meta[f"{group}_action_end"])
            if end <= start:
                raise ValueError(f"Action group {group!r} is empty; check SET action config.")
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
            "set_event_tokens": torch.as_tensor(
                float(self.set_num_event_queries), device=loss_total.device, dtype=loss_total.dtype
            ),
            "set_forward_s": torch.as_tensor(
                time.perf_counter() - t0, device=loss_total.device, dtype=loss_total.dtype
            ).detach(),
        }
        if self.set_history_bias_enabled:
            loss_dict["set_history_bias_gate"] = self._set_history_bias_gate_value(
                device=loss_total.device,
                dtype=loss_total.dtype,
            ).detach()
        for name, value in set_metrics.items():
            loss_dict[name] = value.detach()
        for group, group_loss in action_group_losses.items():
            loss_dict[f"loss_action_{group}"] = (self.loss_lambda_action * group_loss).detach()
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_action_noise_with_set_cache(
        self,
        *,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        event_prefix: torch.Tensor,
        history_event_prefix_for_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre_with_event = self._concat_set_action_pre(
            action_pre,
            event_prefix,
            include_event_query=True,
            history_event_prefix_for_bias=history_event_prefix_for_bias,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre_with_event["tokens"],
            action_freqs=action_pre_with_event["freqs"],
            action_t_mod=action_pre_with_event["t_mod"],
            action_context_payload={
                "context": action_pre_with_event["context"],
                "mask": action_pre_with_event["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        action_meta = action_pre_with_event["meta"]
        pred_start = int(action_meta["prediction_action_start"])
        pred_end = int(action_meta["prediction_action_end"])
        return self.action_expert.post_dit(action_tokens[:, pred_start:pred_end], action_pre)

    @torch.no_grad()
    def infer_action(self, *args, history_video: Optional[torch.Tensor] = None, history_action: Optional[torch.Tensor] = None, **kwargs):
        if not self.set_enabled:
            return super().infer_action(*args, history_video=history_video, history_action=history_action, **kwargs)
        prompt = kwargs.pop("prompt", args[0] if len(args) > 0 else None)
        input_image = kwargs.pop("input_image", args[1] if len(args) > 1 else None)
        action_horizon = kwargs.pop("action_horizon", args[2] if len(args) > 2 else None)
        if len(args) > 3:
            raise TypeError("SETWAM.infer_action accepts at most positional prompt, input_image, action_horizon.")
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

        history_event = self._build_set_history_event(
            history_action=history_action,
            history_video=history_video,
            context=context,
            context_mask=context_mask,
            tiled=tiled,
        )
        full_event_prefix = history_event["prefix"]
        event_prefix = self._set_action_prefix_for_mot(full_event_prefix)
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
        attention_mask = self._build_set_attention_mask(
            video_seq_len=video_seq_len,
            event_prefix_len=event_prefix.shape[1],
            current_action_len=latents_action.shape[1],
            extra_action_len=latents_action.shape[1] if self.extra_prediction_action_tokens else 0,
            event_query_len=self.set_num_event_queries,
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
            pred_action = self._predict_action_noise_with_set_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                event_prefix=event_prefix,
                history_event_prefix_for_bias=full_event_prefix,
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
            "set_state_readout": self.set_state_readout.state_dict(),
            "set_event_encoder": self.set_event_encoder.state_dict(),
            "set_transport": self.set_transport.state_dict(),
            "set_query_to_alpha": self.set_query_to_alpha.state_dict(),
            "set_current_event_query": self.set_current_event_query.detach().cpu(),
            "loss_lambda_action_psd": self.loss_lambda_action_psd,
            "loss_lambda_semantic_alignment": self.loss_lambda_semantic_alignment,
            "loss_lambda_set": self.loss_lambda_set,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if getattr(self, "set_history_bias_projector", None) is not None:
            payload["set_history_bias_projector"] = self.set_history_bias_projector.state_dict()
            payload["set_history_bias_gate"] = self.set_history_bias_gate.detach().cpu()
        if self.semantic_alignment_adapter is not None:
            payload["semantic_alignment_adapter"] = self.semantic_alignment_adapter.state_dict()
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        for key, module in (
            ("set_state_readout", self.set_state_readout),
            ("set_event_encoder", self.set_event_encoder),
            ("set_transport", self.set_transport),
            ("set_query_to_alpha", self.set_query_to_alpha),
        ):
            if key in payload:
                result = module.load_state_dict(payload[key], strict=False)
                _log_load_state_dict_result(key, result, str(path))
        if getattr(self, "set_history_bias_projector", None) is not None and "set_history_bias_projector" in payload:
            result = self.set_history_bias_projector.load_state_dict(
                payload["set_history_bias_projector"], strict=False
            )
            _log_load_state_dict_result("set_history_bias_projector", result, str(path))
        if hasattr(self, "set_history_bias_gate") and "set_history_bias_gate" in payload:
            self.set_history_bias_gate.data.copy_(
                payload["set_history_bias_gate"].to(
                    device=self.set_history_bias_gate.device,
                    dtype=self.set_history_bias_gate.dtype,
                )
            )
        if "set_current_event_query" in payload:
            self.set_current_event_query.data.copy_(
                payload["set_current_event_query"].to(
                    device=self.set_current_event_query.device,
                    dtype=self.set_current_event_query.dtype,
                )
            )
        return payload


def create_aed_core(
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

    return SETWAM.from_wan22_pretrained(
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
