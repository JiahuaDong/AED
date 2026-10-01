import logging
import math
import os
import sys
import time
import inspect
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wam.datasets.lerobot.processors.wam_processor import WAMProcessor
from wam.datasets.lerobot.robotwin_camera import compose_robotwin_cameras
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
    sim_model: Optional[str],
    extra_overrides: Optional[list[str]] = None,
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")
    if not _is_none_like(sim_model):
        overrides.append(f"model={str(sim_model)}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _format_hydra_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _collect_cfg_overrides(usr_args: Dict[str, Any], prefix: str) -> list[str]:
    overrides = []
    dotted_prefix = f"{prefix}."
    for key, value in usr_args.items():
        if key.startswith(dotted_prefix):
            overrides.append(f"{key}={_format_hydra_override_value(value)}")
    return overrides


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _find_trained_config(ckpt_path: Path) -> Optional[Path]:
    """Find a saved training config near a checkpoint, including a path pointer."""
    for parent in list(ckpt_path.resolve().parents)[:5]:
        for name in ("config.yaml", "resolved_train_config.yaml"):
            candidate = parent / name
            if candidate.is_file():
                return candidate.resolve()
        pointer = parent / "training_config_path.txt"
        if pointer.is_file():
            value = pointer.read_text(encoding="utf-8").strip()
            candidate = Path(os.path.expandvars(os.path.expanduser(value)))
            if not candidate.is_absolute():
                candidate = pointer.parent / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Training config pointer is invalid: {pointer} -> {candidate}")
            return candidate
    return None


def _apply_explicit_model_overrides(cfg: DictConfig, usr_args: Dict[str, Any]) -> None:
    for key, value in usr_args.items():
        if key.startswith("model."):
            OmegaConf.update(cfg, key, value, merge=True, force_add=True)


class WorldActionRobotWinPolicy:
    @staticmethod
    def _validate_grouped_history_alignment(
        *,
        action_horizon: int,
        group_size: int,
        action_video_freq_ratio: int,
        image_stride: int,
        history_video_frames: int,
    ) -> None:
        if group_size <= 0 or action_horizon % group_size != 0:
            raise ValueError(
                "Grouped history inference requires action horizon divisible by K, got "
                f"horizon={action_horizon}, K={group_size}."
            )
        if group_size != action_video_freq_ratio:
            raise ValueError(
                "History action grouping K must equal data action_video_freq_ratio, got "
                f"K={group_size}, ratio={action_video_freq_ratio}."
            )
        if group_size != image_stride:
            raise ValueError(
                "History action grouping K must equal online image sampling stride, got "
                f"K={group_size}, stride={image_stride}."
            )
        expected_intervals = action_horizon // group_size
        if expected_intervals != history_video_frames:
            raise ValueError(
                "Grouped history intervals must equal configured history video frames, got "
                f"{expected_intervals} and {history_video_frames}."
            )

    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        action_video_freq_ratio: int,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: WAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self._action_video_freq_ratio = max(1, int(action_video_freq_ratio))

        infer_params = inspect.signature(self.model.infer_action).parameters
        self._uses_history_inputs = "history_video" in infer_params and "history_action" in infer_params
        history_adapter = getattr(self.model, "history_adapter", None)
        history_requires_start_state = bool(
            getattr(history_adapter, "action_sum_enabled", False)
            and getattr(history_adapter, "action_preprocess_type", "legacy_sum") == "relative"
        )
        self._uses_history_state = (
            self._uses_history_inputs
            and history_requires_start_state
            and "history_state" in infer_params
        )
        self._history_ablation_mode = os.environ.get("WAM_HISTORY_ABLATION", "real").strip().lower()
        valid_history_modes = {"real", "zero_all", "video_only"}
        if self._history_ablation_mode not in valid_history_modes:
            raise ValueError(
                "Unsupported WAM_HISTORY_ABLATION="
                f"{self._history_ablation_mode!r}; expected one of {sorted(valid_history_modes)}."
            )
        self._history_video_frames = int(
            getattr(self.model, "history_frames", max(0, int(getattr(self.model, "history_video_frames", 1)) - 1))
        )
        self._history_action_horizon = int(
            getattr(self.model, "history_action_horizon", None) or self.action_horizon
        )
        if self._history_video_frames > 0:
            if self._history_action_horizon % self._history_video_frames != 0:
                raise ValueError(
                    "History action horizon must be divisible by history video frames for aligned sampling, got "
                    f"{self._history_action_horizon} and {self._history_video_frames}."
                )
            self._history_image_stride = self._history_action_horizon // self._history_video_frames
        else:
            self._history_image_stride = 1
        if getattr(history_adapter, "action_one_token_per_group", False):
            group_size = int(getattr(history_adapter, "action_sum_group_size", 0))
            self._validate_grouped_history_alignment(
                action_horizon=self._history_action_horizon,
                group_size=group_size,
                action_video_freq_ratio=self._action_video_freq_ratio,
                image_stride=self._history_image_stride,
                history_video_frames=self._history_video_frames,
            )
        self._action_dim = int(getattr(getattr(self.model, "action_expert", None), "action_dim", 0))
        if self._uses_history_inputs and self._action_dim <= 0:
            raise ValueError("History-aware inference requires model.action_expert.action_dim > 0.")

        self._action_delta_scale = float(os.environ.get("WAM_ACTION_DELTA_SCALE", "1.0"))
        if not math.isfinite(self._action_delta_scale) or self._action_delta_scale <= 0.0:
            raise ValueError(
                "WAM_ACTION_DELTA_SCALE must be finite and positive, got "
                f"{self._action_delta_scale}."
            )

        chunk_ensemble_mode = os.environ.get("WAM_CHUNK_ENSEMBLE", "off").strip().lower()
        if chunk_ensemble_mode in {"0", "false", "no", "none"}:
            chunk_ensemble_mode = "off"
        elif chunk_ensemble_mode in {"1", "true", "yes"}:
            chunk_ensemble_mode = "exp"
        valid_chunk_ensemble_modes = {"off", "uniform", "exp"}
        if chunk_ensemble_mode not in valid_chunk_ensemble_modes:
            raise ValueError(
                "Unsupported WAM_CHUNK_ENSEMBLE="
                f"{chunk_ensemble_mode!r}; expected one of {sorted(valid_chunk_ensemble_modes)}."
            )
        self._chunk_ensemble_mode = chunk_ensemble_mode
        self._chunk_ensemble_decay = float(os.environ.get("WAM_CHUNK_ENSEMBLE_DECAY", "0.5"))
        if self._chunk_ensemble_decay < 0.0:
            raise ValueError("WAM_CHUNK_ENSEMBLE_DECAY must be non-negative.")
        self._chunk_ensemble_max_chunks = max(1, int(os.environ.get("WAM_CHUNK_ENSEMBLE_MAX_CHUNKS", "4")))

        self.pending_actions: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self.chunk_predictions: deque[dict[str, Any]] = deque(maxlen=self._chunk_ensemble_max_chunks)
        self.history_images: deque[torch.Tensor] = deque(maxlen=max(0, self._history_action_horizon))
        self.history_actions: deque[np.ndarray] = deque(maxlen=max(1, self._history_action_horizon))
        self.history_states: deque[np.ndarray] = deque(maxlen=max(1, self._history_action_horizon))
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d | history=%s | history_ablation=%s | chunk_ensemble=%s decay=%.3g max_chunks=%d | action_delta_scale=%.3g",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self._uses_history_inputs,
            self._history_ablation_mode,
            self._chunk_ensemble_mode,
            self._chunk_ensemble_decay,
            self._chunk_ensemble_max_chunks,
            self._action_delta_scale,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2:
            raise ValueError(f"Expected action array [T,D], got {action.shape}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        normalized = normalizer.forward(torch.as_tensor(action, dtype=torch.float32))
        return normalized.numpy()

    @staticmethod
    def _gripper_indices(action_dim: int) -> tuple[int, ...]:
        if action_dim > 0 and action_dim % 7 == 0:
            return tuple(7 * arm + 6 for arm in range(action_dim // 7))
        if action_dim > 0 and action_dim % 8 == 0:
            return tuple(8 * arm + 7 for arm in range(action_dim // 8))
        return ()

    def _scale_absolute_action_delta(
        self,
        action_chunk: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        current_state = np.asarray(current_state, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk [T,D], got {action_chunk.shape}")
        if current_state.shape != (action_chunk.shape[1],):
            raise ValueError(
                "Current state must match action dimension, got "
                f"state={current_state.shape}, action={action_chunk.shape}."
            )
        if self._action_delta_scale == 1.0:
            return action_chunk

        scaled = current_state[None, :] + self._action_delta_scale * (
            action_chunk - current_state[None, :]
        )
        gripper_indices = self._gripper_indices(action_chunk.shape[1])
        if gripper_indices:
            scaled[:, gripper_indices] = action_chunk[:, gripper_indices]
        return scaled.astype(np.float32, copy=False)

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        cameras = torch.stack([
            torch.from_numpy(np.asarray(obs_data[name]["rgb"], dtype=np.uint8)).permute(2, 0, 1)
            for name in ("head_camera", "left_camera", "right_camera")
        ]).to(dtype=torch.float32).div_(255.0)
        image_tensor = compose_robotwin_cameras(cameras).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor.mul_(2.0).sub_(1.0)
        return image_tensor

    def _image_to_history_cpu(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError(f"Expected image tensor [1,3,H,W], got {tuple(image_tensor.shape)}")
        return image_tensor[0].detach().to(device="cpu", dtype=torch.float32)

    def _build_history_video_tensor(self, current_image_cpu: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._uses_history_inputs:
            return None
        if self._history_video_frames <= 0:
            return None
        if self._history_ablation_mode == "zero_all":
            history_frames = [current_image_cpu] * self._history_video_frames
            return torch.stack([frame.to(dtype=torch.float32) for frame in history_frames], dim=1).unsqueeze(0)
        action_rate_frames = list(self.history_images)[-self._history_action_horizon :]
        if len(action_rate_frames) < self._history_action_horizon:
            pad_frame = action_rate_frames[0] if action_rate_frames else current_image_cpu
            action_rate_frames = [pad_frame] * (
                self._history_action_horizon - len(action_rate_frames)
            ) + action_rate_frames
        history_frames = action_rate_frames[:: self._history_image_stride]
        if len(history_frames) != self._history_video_frames:
            raise RuntimeError(
                "Aligned history image sampling produced an unexpected length: "
                f"expected {self._history_video_frames}, got {len(history_frames)}."
            )
        return torch.stack([frame.to(dtype=torch.float32) for frame in history_frames], dim=1).unsqueeze(0)

    def _build_history_action_tensor(self) -> Optional[torch.Tensor]:
        if not self._uses_history_inputs:
            return None
        history = torch.zeros(
            (self._history_action_horizon, self._action_dim),
            dtype=torch.float32,
        )
        if self._history_ablation_mode in {"zero_all", "video_only"}:
            return history.unsqueeze(0)
        actions = list(self.history_actions)[-self._history_action_horizon :]
        if actions:
            action_tensor = torch.as_tensor(np.stack(actions, axis=0), dtype=torch.float32)
            if action_tensor.shape[1] != self._action_dim:
                raise ValueError(
                    f"History action dim mismatch: expected {self._action_dim}, got {action_tensor.shape[1]}"
                )
            history[-action_tensor.shape[0] :] = action_tensor
        return history.unsqueeze(0)

    def _build_history_state_tensor(self, current_state: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._uses_history_state:
            return None
        current_state = current_state.detach().to(device="cpu", dtype=torch.float32)
        if current_state.ndim == 2 and current_state.shape[0] == 1:
            current_state = current_state[0]
        if current_state.shape != (self._action_dim,):
            raise ValueError(
                f"Expected normalized history state shape {(self._action_dim,)}, got {tuple(current_state.shape)}"
            )
        if self._history_ablation_mode in {"zero_all", "video_only"}:
            return current_state.view(1, 1, -1).expand(1, self._history_action_horizon + 1, -1).contiguous()
        if len(self.history_states) != len(self.history_actions):
            raise RuntimeError(
                "History state/action cache lost temporal alignment: "
                f"states={len(self.history_states)}, actions={len(self.history_actions)}."
            )

        states = list(self.history_states)[-self._history_action_horizon :] + [current_state.numpy()]
        if len(states) < self._history_action_horizon + 1:
            pad_state = states[0] if states else current_state.numpy()
            states = [pad_state] * (self._history_action_horizon + 1 - len(states)) + states
        return torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32).unsqueeze(0)

    def _record_history_image(self, image_cpu: Optional[torch.Tensor]) -> None:
        if not self._uses_history_inputs or image_cpu is None or self._history_video_frames <= 0:
            return
        if self._history_ablation_mode == "zero_all":
            return
        self.history_images.append(image_cpu.detach().to(device="cpu", dtype=torch.float32))

    def _record_history_action(self, action_norm: np.ndarray) -> None:
        if not self._uses_history_inputs:
            return
        if self._history_ablation_mode in {"zero_all", "video_only"}:
            return
        action_norm = np.asarray(action_norm, dtype=np.float32)
        if action_norm.shape != (self._action_dim,):
            raise ValueError(f"Expected normalized action shape {(self._action_dim,)}, got {action_norm.shape}")
        self.history_actions.append(action_norm.copy())

    def _record_history_state(self, state_norm: Optional[torch.Tensor]) -> None:
        if not self._uses_history_state:
            return
        if self._history_ablation_mode in {"zero_all", "video_only"}:
            return
        if state_norm is None:
            raise RuntimeError("History-aware inference requires the observation state before every executed action.")
        state_norm = state_norm.detach().to(device="cpu", dtype=torch.float32)
        if state_norm.ndim == 2 and state_norm.shape[0] == 1:
            state_norm = state_norm[0]
        state_np = state_norm.numpy()
        if state_np.shape != (self._action_dim,):
            raise ValueError(f"Expected normalized state shape {(self._action_dim,)}, got {state_np.shape}")
        self.history_states.append(state_np.copy())

    def _chunk_ensemble_enabled(self) -> bool:
        return self._chunk_ensemble_mode != "off"

    def _prune_chunk_predictions(self) -> None:
        if not self.chunk_predictions:
            return
        current_step = int(self.step_count)
        kept = deque(maxlen=self._chunk_ensemble_max_chunks)
        for record in self.chunk_predictions:
            start = int(record["start"])
            length = int(record["action"].shape[0])
            if start + length > current_step:
                kept.append(record)
        self.chunk_predictions = kept

    def _register_chunk_prediction(self, action_chunk: np.ndarray, action_norm: np.ndarray) -> None:
        if not self._chunk_ensemble_enabled():
            return
        self._prune_chunk_predictions()
        self.chunk_predictions.append(
            {
                "start": int(self.step_count),
                "action": np.asarray(action_chunk, dtype=np.float32).copy(),
                "norm": np.asarray(action_norm, dtype=np.float32).copy(),
            }
        )

    def _ensemble_action_for_step(self, absolute_step: int) -> tuple[np.ndarray, np.ndarray]:
        actions = []
        norms = []
        weights = []
        for record in self.chunk_predictions:
            start = int(record["start"])
            local_step = int(absolute_step) - start
            if local_step < 0 or local_step >= int(record["action"].shape[0]):
                continue
            age_chunks = max(0.0, float(int(self.step_count) - start) / float(max(1, self.replan_steps)))
            if self._chunk_ensemble_mode == "uniform":
                weight = 1.0
            else:
                weight = float(np.exp(-self._chunk_ensemble_decay * age_chunks))
            actions.append(record["action"][local_step])
            norms.append(record["norm"][local_step])
            weights.append(weight)
        if not actions:
            raise RuntimeError("No chunk prediction covers the requested action step.")
        weights_np = np.asarray(weights, dtype=np.float32)
        weights_np = weights_np / np.maximum(weights_np.sum(), 1.0e-8)
        action = np.sum(np.stack(actions, axis=0) * weights_np[:, None], axis=0)
        norm = np.sum(np.stack(norms, axis=0) * weights_np[:, None], axis=0)
        return action.astype(np.float32), norm.astype(np.float32)

    def _infer_action_chunk(
        self,
        observation: Dict[str, Any],
        instruction: str,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        image_tensor = self._build_robotwin_image_tensor(observation)
        current_image_cpu = self._image_to_history_cpu(image_tensor)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        if self._uses_history_inputs:
            infer_kwargs["history_video"] = self._build_history_video_tensor(current_image_cpu)
            infer_kwargs["history_action"] = self._build_history_action_tensor()
            if self._uses_history_state:
                infer_kwargs["history_state"] = self._build_history_state_tensor(proprio)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_norm = action_tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        action_chunk = self._scale_absolute_action_delta(action_chunk, state_vector)
        if self._action_delta_scale != 1.0:
            action_norm = self._normalize_action(action_chunk)
        return action_chunk, action_norm, current_image_cpu, proprio.detach().to(device="cpu", dtype=torch.float32)

    def _fill_action_queue(
        self,
        observation: Dict[str, Any],
        instruction: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_chunk, action_norm, current_image_cpu, current_state_cpu = self._infer_action_chunk(
            observation=observation,
            instruction=instruction,
        )
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        if self._chunk_ensemble_enabled():
            self._register_chunk_prediction(action_chunk, action_norm)
            for i in range(n_exec):
                action_i, action_norm_i = self._ensemble_action_for_step(int(self.step_count) + i)
                self.pending_actions.append((action_i, action_norm_i))
        else:
            for i in range(n_exec):
                self.pending_actions.append(
                    (
                        np.asarray(action_chunk[i], dtype=np.float32),
                        np.asarray(action_norm[i], dtype=np.float32),
                    )
                )
        return current_image_cpu, current_state_cpu

    def should_request_observation(self) -> bool:
        if not self.pending_actions:
            return True
        if not self._uses_history_inputs:
            return False
        if self._history_ablation_mode == "zero_all":
            return False
        return True

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        current_image_cpu = None
        current_state_cpu = None
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for wam)."
                )
            instruction = task_env.get_instruction()
            current_image_cpu, current_state_cpu = self._fill_action_queue(
                observation=observation,
                instruction=instruction,
            )
        elif self._uses_history_inputs and observation is not None:
            current_image_cpu = self._image_to_history_cpu(self._build_robotwin_image_tensor(observation))
            state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
            current_state_cpu = self._normalize_state(state_vector).detach().to(device="cpu", dtype=torch.float32)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action, action_norm = self.pending_actions.popleft()
        self._record_history_image(current_image_cpu)
        self._record_history_state(current_state_cpu)
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self._record_history_action(action_norm)
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.chunk_predictions.clear()
        self.history_images.clear()
        self.history_actions.clear()
        self.history_states.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    sim_model = usr_args.get("sim_model")
    cfg_overrides = _collect_cfg_overrides(usr_args, "model")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
        sim_model=sim_model,
        extra_overrides=cfg_overrides,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    trained_cfg_path = _find_trained_config(Path(str(checkpoint_path)))
    require_trained_cfg = _parse_bool(os.environ.get("WAM_REQUIRE_TRAIN_CONFIG", "false"))
    if trained_cfg_path is None:
        if require_trained_cfg:
            raise FileNotFoundError(f"No training config found near checkpoint: {checkpoint_path}")
        logger.warning("No training config found near %s; using task preset", checkpoint_path)
    else:
        trained_cfg = OmegaConf.load(trained_cfg_path)
        if "model" not in trained_cfg or "data" not in trained_cfg or "train" not in trained_cfg.data:
            raise ValueError(f"Training config lacks model/data.train: {trained_cfg_path}")
        cfg.model = trained_cfg.model
        cfg.data.train.processor = trained_cfg.data.train.processor
        _apply_explicit_model_overrides(cfg, usr_args)
        print(f"[WAM] Loaded model + processor from trained config: {trained_cfg_path}", flush=True)
        print(
            "[WAM] Resolved target="
            f"{OmegaConf.to_container(cfg.model.aed.target, resolve=True)} "
            f"processor={cfg.data.train.processor.get('_target_', '?')}",
            flush=True,
        )

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        action_video_freq_ratio=int(cfg.data.train.action_video_freq_ratio),
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
