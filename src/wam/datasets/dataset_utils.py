
from typing import Any, Optional

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image


def obtain_image_size(data: torch.Tensor | Image.Image) -> tuple[int, int]:
    r"""Return spatial size from a PIL image or image/video tensor.

    Args:
        data (torch.Tensor | Image.Image): Input image or video tensor.
    Returns:
        width (int): Input width.
        height (int): Input height.
    """

    if isinstance(data, Image.Image):
        width, height = data.size
    elif isinstance(data, torch.Tensor):
        height, width = data.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height

class ResizeSmallestSideAspectPreserving:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        r"""Resize while preserving aspect ratio.

        The output is scaled so both spatial dimensions are at least the
        requested target size.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor | Image.Image: Resized image or video tensor.
        """

        assert self.args is not None, "Please specify args in augmentations"

        img_w, img_h = self.args["img_w"], self.args["img_h"]

        orig_w, orig_h = obtain_image_size(video)
        scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
        target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

        assert (
            target_size[0] >= img_h and target_size[1] >= img_w
        ), f"Resize error. orig {(orig_w, orig_h)} desire {(img_w, img_h)} compute {target_size}"

        return transforms_F.resize(
            video,
            size=target_size,  # type: ignore
            interpolation=self.args.get("interpolation", transforms_F.InterpolationMode.BICUBIC),
            antialias=True,
        )


class CenterCrop:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        r"""Center crop to the requested spatial size.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor | Image.Image: Center cropped image or video tensor.
        """
        assert (
            (self.args is not None) and ("img_w" in self.args) and ("img_h" in self.args)
        ), "Please specify size in args"

        img_w, img_h = self.args["img_w"], self.args["img_h"]
        return transforms_F.center_crop(video, [img_h, img_w])


class SynchronizedVideoAugmentation:
    """Apply temporally consistent video augmentation with an optional clean branch.

    Inputs may be ``[T, C, H, W]`` or ``[N, T, C, H, W]``. Parameters are
    shared over time. By default they are also shared across cameras for
    backward compatibility; ImageWAM-style configs can sample each camera
    independently while retaining temporal consistency.
    """

    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        if args is None:
            raise ValueError("Please specify synchronized video augmentation args")

        self.p = float(args.get("p", 1.0))
        self.synchronize_across_cameras = bool(
            args.get("synchronize_across_cameras", True)
        )
        augment_types = args.get("augment_types", ("both",))
        if not isinstance(augment_types, (list, tuple)) or not augment_types:
            raise ValueError("augment_types must be a non-empty list or tuple")
        self.augment_types = tuple(str(value) for value in augment_types)
        valid_augment_types = {"corrupt_only", "color_only", "both"}
        invalid_augment_types = set(self.augment_types) - valid_augment_types
        if invalid_augment_types:
            raise ValueError(
                f"Unsupported augment_types: {sorted(invalid_augment_types)}"
            )

        self.crop_scale = float(args.get("crop_scale", 1.0))
        crop_scale_range = args.get(
            "crop_scale_range", (self.crop_scale, self.crop_scale)
        )
        if not isinstance(crop_scale_range, (list, tuple)) or len(crop_scale_range) != 2:
            raise ValueError(
                "crop_scale_range must be a two-element list or tuple, "
                f"got {crop_scale_range!r}"
            )
        self.crop_scale_range = tuple(float(value) for value in crop_scale_range)
        self.rotation_degrees = float(args.get("rotation_degrees", 0.0))
        self.brightness = float(args.get("brightness", 0.0))
        self.contrast = float(args.get("contrast", 0.0))
        self.saturation = float(args.get("saturation", 0.0))
        self.hue = float(args.get("hue", 0.0))
        gamma_range = args.get("gamma_range", (1.0, 1.0))
        if not isinstance(gamma_range, (list, tuple)) or len(gamma_range) != 2:
            raise ValueError(
                f"gamma_range must be a two-element list or tuple, got {gamma_range!r}"
            )
        self.gamma_range = tuple(float(value) for value in gamma_range)
        exposure_ev_range = args.get("exposure_ev_range", (0.0, 0.0))
        if not isinstance(exposure_ev_range, (list, tuple)) or len(exposure_ev_range) != 2:
            raise ValueError(
                "exposure_ev_range must be a two-element list or tuple, "
                f"got {exposure_ev_range!r}"
            )
        self.exposure_ev_range = tuple(float(value) for value in exposure_ev_range)
        self.gaussian_noise_std = float(args.get("gaussian_noise_std", 0.0))
        self.gaussian_blur_probability = float(
            args.get("gaussian_blur_probability", 0.0)
        )
        self.gaussian_blur_kernel_size = int(args.get("gaussian_blur_kernel_size", 5))
        gaussian_blur_sigma = args.get("gaussian_blur_sigma", (0.1, 1.0))
        if not isinstance(gaussian_blur_sigma, (list, tuple)) or len(gaussian_blur_sigma) != 2:
            raise ValueError(
                "gaussian_blur_sigma must be a two-element list or tuple, "
                f"got {gaussian_blur_sigma!r}"
            )
        self.gaussian_blur_sigma = tuple(float(value) for value in gaussian_blur_sigma)
        self.fill = args.get("fill", 0.5)

        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        if not 0.0 < self.crop_scale <= 1.0:
            raise ValueError(f"crop_scale must be in (0, 1], got {self.crop_scale}")
        if not (
            0.0 < self.crop_scale_range[0] <= self.crop_scale_range[1] <= 1.0
        ):
            raise ValueError(
                "crop_scale_range must satisfy 0 < min <= max <= 1, "
                f"got {self.crop_scale_range}"
            )
        if self.rotation_degrees < 0.0:
            raise ValueError(
                f"rotation_degrees must be non-negative, got {self.rotation_degrees}"
            )
        for name, value in (
            ("brightness", self.brightness),
            ("contrast", self.contrast),
            ("saturation", self.saturation),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 <= self.hue <= 0.5:
            raise ValueError(f"hue must be in [0, 0.5], got {self.hue}")
        if not 0.0 < self.gamma_range[0] <= self.gamma_range[1]:
            raise ValueError(
                f"gamma_range must satisfy 0 < min <= max, got {self.gamma_range}"
            )
        if self.exposure_ev_range[0] > self.exposure_ev_range[1]:
            raise ValueError(
                "exposure_ev_range must satisfy min <= max, "
                f"got {self.exposure_ev_range}"
            )
        if self.gaussian_noise_std < 0.0:
            raise ValueError(
                f"gaussian_noise_std must be non-negative, got {self.gaussian_noise_std}"
            )
        if not 0.0 <= self.gaussian_blur_probability <= 1.0:
            raise ValueError(
                "gaussian_blur_probability must be in [0, 1], "
                f"got {self.gaussian_blur_probability}"
            )
        if (
            self.gaussian_blur_kernel_size <= 0
            or self.gaussian_blur_kernel_size % 2 == 0
        ):
            raise ValueError(
                "gaussian_blur_kernel_size must be a positive odd integer, "
                f"got {self.gaussian_blur_kernel_size}"
            )
        if (
            self.gaussian_blur_sigma[0] <= 0.0
            or self.gaussian_blur_sigma[1] < self.gaussian_blur_sigma[0]
        ):
            raise ValueError(
                "gaussian_blur_sigma must satisfy 0 < min <= max, "
                f"got {self.gaussian_blur_sigma}"
            )
        if self.fill != "mean":
            self.fill = float(self.fill)
            if not 0.0 <= self.fill <= 1.0:
                raise ValueError(f"fill must be in [0, 1] or 'mean', got {self.fill}")

    @staticmethod
    def _uniform(center: float, radius: float) -> float:
        if radius == 0.0:
            return center
        return float(torch.empty(()).uniform_(center - radius, center + radius).item())

    @staticmethod
    def _uniform_range(bounds: tuple[float, float]) -> float:
        low, high = bounds
        if low == high:
            return low
        return float(torch.empty(()).uniform_(low, high).item())

    def _sample_camera_params(self, video: torch.Tensor) -> dict[str, Any]:
        height, width = video.shape[-2:]
        crop_scale = self._uniform_range(self.crop_scale_range)
        crop_height = max(1, min(height, int(round(height * crop_scale))))
        crop_width = max(1, min(width, int(round(width * crop_scale))))
        max_top = height - crop_height
        max_left = width - crop_width
        top = int(torch.randint(max_top + 1, ()).item()) if max_top > 0 else 0
        left = int(torch.randint(max_left + 1, ()).item()) if max_left > 0 else 0
        augment_type_index = int(torch.randint(len(self.augment_types), ()).item())

        return {
            "input_height": int(height),
            "input_width": int(width),
            "crop_height": crop_height,
            "crop_width": crop_width,
            "top": top,
            "left": left,
            "augment_type": self.augment_types[augment_type_index],
            "angle": self._uniform(0.0, self.rotation_degrees),
            "brightness_factor": self._uniform(1.0, self.brightness),
            "contrast_factor": self._uniform(1.0, self.contrast),
            "saturation_factor": self._uniform(1.0, self.saturation),
            "hue_factor": self._uniform(0.0, self.hue),
            "gamma": self._uniform_range(self.gamma_range),
            "exposure_ev": self._uniform_range(self.exposure_ev_range),
            "color_order": torch.randperm(4).tolist(),
            "apply_gaussian_blur": bool(
                torch.rand(()).item() < self.gaussian_blur_probability
            ),
            "gaussian_blur_sigma": self._uniform(
                sum(self.gaussian_blur_sigma) / 2.0,
                (self.gaussian_blur_sigma[1] - self.gaussian_blur_sigma[0]) / 2.0,
            ),
            "gaussian_noise_seed": int(
                torch.randint(0, torch.iinfo(torch.int32).max, ()).item()
            ),
        }

    def sample_params(self, video: torch.Tensor) -> dict[str, Any]:
        if video.ndim not in {4, 5}:
            raise ValueError(
                "SynchronizedVideoAugmentation expects [T,C,H,W] or [N,T,C,H,W], "
                f"got {tuple(video.shape)}"
            )
        num_cameras = int(video.shape[0]) if video.ndim == 5 else 1
        apply = bool(torch.rand(()).item() < self.p)
        camera_params: list[dict[str, Any]] = []
        if apply:
            shared = self._sample_camera_params(video[0] if video.ndim == 5 else video)
            camera_params.append(shared)
            for camera_index in range(1, num_cameras):
                if self.synchronize_across_cameras:
                    camera_params.append(shared)
                else:
                    camera_params.append(self._sample_camera_params(video[camera_index]))
        return {
            "apply": apply,
            "num_cameras": num_cameras,
            "camera_params": camera_params,
        }

    def _apply_camera_params(
        self,
        video: torch.Tensor,
        params: dict[str, Any],
    ) -> torch.Tensor:
        expected_size = (int(params["input_height"]), int(params["input_width"]))
        if tuple(video.shape[-2:]) != expected_size:
            raise ValueError(
                "Cannot reuse synchronized augmentation params across different image sizes: "
                f"expected {expected_size}, got {tuple(video.shape[-2:])}"
            )

        original_shape = video.shape
        flat = video.reshape(-1, *video.shape[-3:])
        # Match ImageWAM exactly: appearance/corruption first, then geometry.
        augment_type = str(params["augment_type"])
        if augment_type in {"color_only", "both"}:
            color_ops = (
                lambda x: transforms_F.adjust_brightness(x, float(params["brightness_factor"])),
                lambda x: transforms_F.adjust_contrast(x, float(params["contrast_factor"])),
                lambda x: transforms_F.adjust_saturation(x, float(params["saturation_factor"])),
                lambda x: transforms_F.adjust_hue(x.clamp(0.0, 1.0), float(params["hue_factor"])),
            )
            for op_index in params["color_order"]:
                flat = color_ops[int(op_index)](flat)
            flat = transforms_F.adjust_gamma(
                flat.clamp(0.0, 1.0), gamma=float(params["gamma"])
            )
            flat = flat * (2.0 ** float(params["exposure_ev"]))
        if augment_type in {"corrupt_only", "both"}:
            if self.gaussian_noise_std > 0.0:
                generator = torch.Generator(device=flat.device)
                generator.manual_seed(int(params["gaussian_noise_seed"]))
                noise = torch.randn(
                    (1,) + tuple(flat.shape[1:]),
                    generator=generator,
                    device=flat.device,
                    dtype=flat.dtype,
                ) * self.gaussian_noise_std
                flat = flat + noise
            if bool(params["apply_gaussian_blur"]):
                flat = transforms_F.gaussian_blur(
                    flat,
                    kernel_size=[
                        self.gaussian_blur_kernel_size,
                        self.gaussian_blur_kernel_size,
                    ],
                    sigma=[
                        float(params["gaussian_blur_sigma"]),
                        float(params["gaussian_blur_sigma"]),
                    ],
                )

        flat = transforms_F.crop(
            flat,
            top=int(params["top"]),
            left=int(params["left"]),
            height=int(params["crop_height"]),
            width=int(params["crop_width"]),
        )
        flat = transforms_F.resize(
            flat,
            size=list(expected_size),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        rotation_fill = float(flat.mean().item()) if self.fill == "mean" else self.fill
        flat = transforms_F.rotate(
            flat,
            angle=float(params["angle"]),
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            expand=False,
            fill=rotation_fill,
        )

        return flat.clamp(0.0, 1.0).reshape(original_shape)

    def __call__(
        self,
        video: torch.Tensor,
        params: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if video.ndim not in {4, 5}:
            raise ValueError(
                "SynchronizedVideoAugmentation expects [T,C,H,W] or [N,T,C,H,W], "
                f"got {tuple(video.shape)}"
            )
        if not video.is_floating_point():
            raise TypeError(
                "SynchronizedVideoAugmentation expects floating point images in [0, 1], "
                f"got dtype={video.dtype}"
            )
        if params is None:
            params = self.sample_params(video)
        num_cameras = int(video.shape[0]) if video.ndim == 5 else 1
        if int(params["num_cameras"]) != num_cameras:
            raise ValueError(
                "Cannot reuse augmentation params with a different number of cameras: "
                f"expected {params['num_cameras']}, got {num_cameras}"
            )
        if not bool(params["apply"]):
            return video, params

        camera_params = params["camera_params"]
        if video.ndim == 4:
            return self._apply_camera_params(video, camera_params[0]), params
        augmented = video.clone()
        for camera_index in range(num_cameras):
            augmented[camera_index] = self._apply_camera_params(
                video[camera_index], camera_params[camera_index]
            )
        return augmented, params


class Normalize:
    def __init__(self, args: Optional[dict] = None) -> None:
        self.args = args

    def __call__(self, video: torch.Tensor | Image.Image) -> torch.Tensor:
        r"""Convert to tensor if needed and normalize by mean/std.

        Args:
            video (torch.Tensor | Image.Image): Input image or video tensor.
        Returns:
            torch.Tensor: Normalized image or video tensor.
        """
        assert self.args is not None, "Please specify args"

        mean = self.args["mean"]
        std = self.args["std"]

        if isinstance(video, torch.Tensor):
            data = video.to(dtype=torch.float32)
            if video.dtype == torch.uint8:
                data = data / 255.0
            data = data.to(dtype=torch.get_default_dtype())
        else:
            data = transforms_F.to_tensor(video)  # division by 255 is applied in to_tensor()

        return transforms_F.normalize(tensor=data, mean=mean, std=std)
