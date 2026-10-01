from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoRAApplySummary:
    wrapped_modules: int
    trainable_parameters: int


class LoRALinear(nn.Module):
    """Linear layer with a zero-initialized LoRA residual branch."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}.")
        if alpha <= 0.0:
            raise ValueError(f"`alpha` must be positive, got {alpha}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}.")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        self.reset_lora_parameters()
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base.bias

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()


def _matches_any(name: str, patterns: Sequence[str] | None) -> bool:
    if not patterns:
        return True
    return any(pattern and pattern in name for pattern in patterns)


def _is_excluded(name: str, patterns: Sequence[str] | None) -> bool:
    if not patterns:
        return False
    return any(pattern and pattern in name for pattern in patterns)


def apply_lora_to_linear_modules(
    root: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
    exclude_modules: Sequence[str] | None = None,
) -> LoRAApplySummary:
    """Replace matching Linear leaves with LoRA-wrapped Linear modules."""

    wrapped = 0
    trainable = 0

    def visit(module: nn.Module, prefix: str):
        nonlocal wrapped, trainable
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                if _matches_any(full_name, target_modules) and not _is_excluded(full_name, exclude_modules):
                    lora = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, child_name, lora)
                    wrapped += 1
                    trainable += sum(param.numel() for param in lora.lora_parameters())
                continue
            visit(child, full_name)

    visit(root, "")
    return LoRAApplySummary(wrapped_modules=wrapped, trainable_parameters=trainable)


def set_lora_trainable_only(root: nn.Module):
    """Freeze a LoRA-wrapped module except LoRA residual parameters."""

    root.requires_grad_(False)
    for module in root.modules():
        if isinstance(module, LoRALinear):
            module.train()
            for parameter in module.lora_parameters():
                parameter.requires_grad_(True)
