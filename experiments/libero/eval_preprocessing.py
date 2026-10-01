"""Image preprocessing and per-task RNG handling for LIBERO evaluation."""
import numpy as np
import torch
from torchvision.transforms import functional as TF


def float_camera(image, height, width):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected RGB uint8 HWC observation')
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    x = x.to(torch.float32) / 255.0
    x = TF.resize(x, [int(height), int(width)],
                  interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    return (x - 0.5) / 0.5


def validate_modes(cfg):
    image = str(cfg.EVALUATION.get('image_preprocessing', 'float32'))
    rng = str(cfg.EVALUATION.get('task_rng_mode', 'continuous'))
    if image not in ('legacy_uint8', 'float32'):
        raise ValueError(f'Unknown image_preprocessing: {image}')
    if rng not in ('legacy_restore', 'continuous'):
        raise ValueError(f'Unknown task_rng_mode: {rng}')
    return image, rng


def seed_at_task_start(mode, runtime):
    if mode not in ('legacy_restore', 'continuous'):
        raise ValueError(mode)
    return mode == 'legacy_restore' or not runtime


def restore_at_task_start(mode, runtime):
    if mode not in ('legacy_restore', 'continuous'):
        raise ValueError(mode)
    return mode == 'legacy_restore' and bool(runtime)
