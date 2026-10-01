"""Shared RoboTwin three-camera composition for training and inference."""

import torch
import torchvision.transforms.functional as transforms_F


def compose_robotwin_cameras(cameras: torch.Tensor) -> torch.Tensor:
    """Compose head/left/right tensors into the trained 384x320 layout.

    ``cameras`` must be shaped ``[3, ..., C, H, W]``. Leading dimensions
    after the camera axis (for example time) are preserved. The resize path is
    intentionally shared by the dataset and online policy so interpolation and
    antialiasing cannot drift.
    """
    if not isinstance(cameras, torch.Tensor):
        raise TypeError(f"cameras must be a torch.Tensor, got {type(cameras)!r}")
    if cameras.ndim < 4 or cameras.shape[0] != 3:
        raise ValueError(f"expected [3,...,C,H,W], got {tuple(cameras.shape)}")
    resized = [
        transforms_F.resize(
            cameras[0],
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms_F.resize(
            cameras[1],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms_F.resize(
            cameras[2],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]
    bottom = torch.cat(resized[1:], dim=-1)
    return torch.cat([resized[0], bottom], dim=-2)
