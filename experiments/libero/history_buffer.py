from collections import deque

import numpy as np
import torch


class LiberoHistoryBuffer:
    """Maintain the exact action-rate history window used by LIBERO training."""

    def __init__(
        self,
        *,
        action_horizon: int,
        history_frames: int,
        action_dim: int,
        action_video_freq_ratio: int,
    ) -> None:
        self.action_horizon = int(action_horizon)
        self.history_frames = int(history_frames)
        self.action_dim = int(action_dim)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if self.action_horizon <= 0 or self.history_frames <= 0 or self.action_dim <= 0:
            raise ValueError(
                "LIBERO history dimensions must be positive, got "
                f"horizon={self.action_horizon}, frames={self.history_frames}, action_dim={self.action_dim}."
            )
        if self.action_video_freq_ratio <= 0:
            raise ValueError(
                f"action_video_freq_ratio must be positive, got {self.action_video_freq_ratio}."
            )
        if self.action_horizon % self.action_video_freq_ratio != 0:
            raise ValueError(
                "History action horizon must be divisible by action_video_freq_ratio, got "
                f"{self.action_horizon} and {self.action_video_freq_ratio}."
            )
        expected_history_frames = self.action_horizon // self.action_video_freq_ratio
        if expected_history_frames != self.history_frames:
            raise ValueError(
                "History video intervals must align with grouped actions, got "
                f"frames={self.history_frames}, expected={expected_history_frames}."
            )

        self._images: deque[torch.Tensor] = deque(maxlen=self.action_horizon)
        self._actions: deque[np.ndarray] = deque(maxlen=self.action_horizon)

    def reset(self) -> None:
        self._images.clear()
        self._actions.clear()

    def record(self, image_before_action: torch.Tensor, normalized_action: np.ndarray) -> None:
        image = image_before_action.detach().to(device="cpu", dtype=torch.float32)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"History image must be [3,H,W], got {tuple(image.shape)}.")
        action = np.asarray(normalized_action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Normalized history action must have shape {(self.action_dim,)}, got {action.shape}."
            )
        self._images.append(image.clone())
        self._actions.append(action.copy())
        if len(self._images) != len(self._actions):
            raise RuntimeError("LIBERO history image/action buffers lost temporal alignment.")

    def build(self, current_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = current_image.detach().to(device="cpu", dtype=torch.float32)
        if current.ndim == 4 and current.shape[0] == 1:
            current = current[0]
        if current.ndim != 3 or current.shape[0] != 3:
            raise ValueError(f"Current image must be [3,H,W], got {tuple(current.shape)}.")

        prior_frames = list(self._images)[-self.action_horizon :]
        frame_sequence = prior_frames + [current]
        if len(frame_sequence) < self.action_horizon + 1:
            first_frame = frame_sequence[0]
            frame_sequence = [first_frame] * (
                self.action_horizon + 1 - len(frame_sequence)
            ) + frame_sequence
        sampled_frames = frame_sequence[:: self.action_video_freq_ratio]
        if len(sampled_frames) != self.history_frames + 1:
            raise RuntimeError(
                "Aligned LIBERO history sampling produced an unexpected number of frames: "
                f"got {len(sampled_frames)}, expected {self.history_frames + 1}."
            )
        history_video = torch.stack(sampled_frames, dim=1).unsqueeze(0)

        history_action = torch.zeros(
            (self.action_horizon, self.action_dim),
            dtype=torch.float32,
        )
        prior_actions = list(self._actions)[-self.action_horizon :]
        if prior_actions:
            action_tensor = torch.as_tensor(np.stack(prior_actions, axis=0), dtype=torch.float32)
            history_action[-action_tensor.shape[0] :] = action_tensor
        return history_video, history_action.unsqueeze(0)

    @property
    def length(self) -> int:
        return len(self._actions)
