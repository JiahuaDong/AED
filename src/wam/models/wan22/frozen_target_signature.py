from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


_DINO_TARGET_TYPES = {"frozen_dino", "dino", "frozen_vision"}
_LINGBOT_TARGET_TYPES = {
    "frozen_lingbot",
    "lingbot",
    "lingbot_vision",
    "frozen_lingbot_vision",
}


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_frozen_target_type(value: Any) -> str:
    normalized = str(value or "frozen_dino").strip().lower()
    if normalized in _LINGBOT_TARGET_TYPES:
        return "frozen_lingbot_vision"
    if normalized in _DINO_TARGET_TYPES:
        return "frozen_dino"
    raise ValueError(f"Unsupported frozen target type for feature signature: {value!r}.")


def canonical_target_token_mode(value: Any) -> str:
    normalized = str(value or "patch").strip().lower()
    if normalized in {"patches", "patch_tokens"}:
        normalized = "patch"
    elif normalized in {"state", "global", "pool", "pooled"}:
        normalized = "pooled"
    if normalized not in {"patch", "pooled"}:
        raise ValueError(f"Unsupported frozen target token mode: {value!r}.")
    return normalized


def frozen_target_feature_spec(
    set_config: Mapping[str, Any],
    *,
    verify_local_model: bool = True,
) -> dict[str, object]:
    """Return the feature-defining signature of a frozen SET target.

    Runtime-only locations are intentionally excluded. For a local LingBot model,
    the actual model.pt digest is included and an optional configured digest is
    verified before any cache/profile can be consumed.
    """

    set_cfg = dict(set_config)
    target_cfg = dict(set_cfg.get("target", {}))
    target_type = canonical_frozen_target_type(target_cfg.get("type", "frozen_dino"))
    default_model_id = (
        "robbyant/lingbot-vision-vit-base"
        if target_type == "frozen_lingbot_vision"
        else "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    spec: dict[str, object] = {
        "target_type": target_type,
        "model_id": str(target_cfg.get("model_id", default_model_id)),
        "feature_dim": int(set_cfg.get("feature_dim", set_cfg.get("feature_hidden_dim", 1024))),
        "image_size": int(target_cfg.get("image_size", 224)),
        "input_range": str(target_cfg.get("input_range", "minus_one_one")).lower(),
        "token_mode": canonical_target_token_mode(target_cfg.get("token_mode", "patch")),
        "pool": str(target_cfg.get("pool", "patch_mean")).lower(),
        "projection_seed": int(target_cfg.get("projection_seed", 0)),
        "normalization_eps": float(set_cfg.get("eps", 1.0e-6)),
    }
    if target_type == "frozen_lingbot_vision":
        spec["variant"] = str(target_cfg.get("variant", "base")).lower()

    configured_digest = target_cfg.get("model_file_sha256")
    local_path = target_cfg.get("local_path")
    if target_type == "frozen_lingbot_vision" and local_path:
        model_file = Path(str(local_path)).expanduser().resolve() / "model.pt"
        if not model_file.is_file():
            raise FileNotFoundError(
                f"LingBot-Vision target signature requires local model.pt: {model_file}."
            )
        if verify_local_model:
            actual_digest = file_sha256(model_file)
            if configured_digest is not None and str(configured_digest) != actual_digest:
                raise ValueError(
                    "LingBot-Vision model.pt SHA256 mismatch: "
                    f"configured={configured_digest}, actual={actual_digest}, path={model_file}."
                )
            spec["model_file_sha256"] = actual_digest
        elif configured_digest is not None:
            spec["model_file_sha256"] = str(configured_digest)
    elif configured_digest is not None:
        spec["model_file_sha256"] = str(configured_digest)

    return spec
