"""Paper-facing WAM entry points.

Naming map: SET/SET-Flare -> Action Experience Dictionary (AED), and
SerialDINOFeaturePredictor -> Motion-Aware Transition (MT) predictor.
The implementation keeps legacy class and state-dict names internally so released
checkpoints remain loadable.
"""
from __future__ import annotations

from typing import Any

from .aed_core import (
    SETWAM,
    SETFrozenLingBotVisionTarget,
    SETFrozenVisionStateTarget,
    create_aed_core,
)
from .aed_transition import (
    HistoryVisualMemoryExtractor,
    SerialDINOFeaturePredictor,
    SerialFeaturePredictionBlock,
    SETFlareWAM,
    create_aed_transition,
)

# Paper terminology. These aliases intentionally preserve the underlying
# Python class and state-dict names for checkpoint compatibility.
ActionExperienceDictionaryWAM = SETWAM
ActionExperienceDictionaryWAMWithMT = SETFlareWAM
VisualHistoryCompressor = HistoryVisualMemoryExtractor
MotionAwareTransitionBlock = SerialFeaturePredictionBlock
MotionAwareTransitionPredictor = SerialDINOFeaturePredictor
FrozenVisualTransitionTarget = SETFrozenVisionStateTarget
FrozenLanguageVisionTarget = SETFrozenLingBotVisionTarget


def create_aed_wam(*args: Any, **kwargs: Any):
    """Build the paper-named AED + MT model from a Hydra config.

    ``aed`` and ``lambda_mt`` are translated to the legacy implementation's
    ``set`` and ``lambda_set`` fields. No model parameters are renamed.
    """
    aed = kwargs.pop("aed", None)
    legacy_set = kwargs.pop("set", None)
    if aed is None:
        aed = legacy_set
    elif legacy_set is not None:
        # A legacy CLI override such as model.set.serial_num_layers=3 is
        # merged into the paper-named AED block instead of being ignored.
        merged = dict(aed)
        merged.update(dict(legacy_set))
        aed = merged
    loss = kwargs.pop("loss", None)
    if loss is not None:
        try:
            loss = dict(loss)
        except Exception:
            pass
        if isinstance(loss, dict) and "lambda_mt" in loss:
            # Keep both keys: the legacy factory ignores lambda_mt, while
            # lambda_set remains available to old validators and checkpoints.
            loss["lambda_set"] = loss["lambda_mt"]
    return create_aed_transition(*args, aed=aed, loss=loss, **kwargs)


def create_aed_baseline_wam(*args: Any, **kwargs: Any):
    """Build the AED module without the MT auxiliary predictor."""
    aed = kwargs.pop("aed", kwargs.pop("set", None))
    loss = kwargs.pop("loss", None)
    if loss is not None:
        try:
            loss = dict(loss)
        except Exception:
            pass
        if isinstance(loss, dict) and "lambda_mt" in loss:
            # Keep both keys: the legacy factory ignores lambda_mt, while
            # lambda_set remains available to old validators and checkpoints.
            loss["lambda_set"] = loss["lambda_mt"]
    return create_aed_core(*args, aed=aed, loss=loss, **kwargs)

__all__ = [
    "ActionExperienceDictionaryWAM",
    "ActionExperienceDictionaryWAMWithMT",
    "VisualHistoryCompressor",
    "MotionAwareTransitionBlock",
    "MotionAwareTransitionPredictor",
    "FrozenVisualTransitionTarget",
    "FrozenLanguageVisionTarget",
    "create_aed_wam",
    "create_aed_baseline_wam",
]
