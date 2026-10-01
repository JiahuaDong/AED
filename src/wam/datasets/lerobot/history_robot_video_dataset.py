import hashlib
import json
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .history_base_lerobot_dataset import HistoryBaseLerobotDataset
from .vision_feature_cache import VisionFeatureCache
from .robotwin_camera import compose_robotwin_cameras
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
    SynchronizedVideoAugmentation,
)
from wam.utils.logging_config import get_logger
from wam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class HistoryRobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        episode_sample_ratio=None,
        episode_sample_max_per_task=None,
        episode_sample_max_per_instruction=None,
        episode_sample_min_per_task=1,
        episode_sample_group_size=None,
        episode_sample_group_key: Optional[str] = "legacy",
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        video_backend: Optional[str] = None,
        video_augmentation: Optional[dict] = None,
        qwen_instruction_cache_path: Optional[str] = None,
        qwen_instruction_missing: str = "error",
        semantic_text_target: str = "task",
        semantic_text_fallback_to_prompt: bool = True,
        vision_feature_cache_dir: Optional[str] = None,
        vision_feature_cache_skip_image_decode: bool = False,
        vision_feature_cache_feature_spec: Optional[dict] = None,
        strict_sample_errors: bool = False,
        gripper_action_convention: str = "dataset_zero_one",
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_sample_ratio=episode_sample_ratio,
            episode_sample_max_per_task=episode_sample_max_per_task,
            episode_sample_max_per_instruction=episode_sample_max_per_instruction,
            episode_sample_min_per_task=episode_sample_min_per_task,
            episode_sample_group_size=episode_sample_group_size,
            episode_sample_group_key=episode_sample_group_key,
            video_backend=video_backend,
        )
        self.history_lerobot_dataset = HistoryBaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            past_obs_size=num_frames - 1,
            past_action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_sample_ratio=episode_sample_ratio,
            episode_sample_max_per_task=episode_sample_max_per_task,
            episode_sample_max_per_instruction=episode_sample_max_per_instruction,
            episode_sample_min_per_task=episode_sample_min_per_task,
            episode_sample_group_size=episode_sample_group_size,
            episode_sample_group_key=episode_sample_group_key,
            video_backend=video_backend,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)
        self.history_lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.qwen_instruction_missing = str(qwen_instruction_missing).lower()
        if self.qwen_instruction_missing not in {"error", "original"}:
            raise ValueError(
                "`qwen_instruction_missing` must be either 'error' or 'original', "
                f"got {qwen_instruction_missing!r}."
            )
        self.semantic_text_target = str(semantic_text_target).lower()
        if self.semantic_text_target not in {"task", "prompt", "none"}:
            raise ValueError(
                "`semantic_text_target` must be one of {'task', 'prompt', 'none'}, "
                f"got {semantic_text_target!r}."
            )
        self.semantic_text_fallback_to_prompt = bool(semantic_text_fallback_to_prompt)
        self.strict_sample_errors = bool(strict_sample_errors)
        self.gripper_action_convention = str(gripper_action_convention).strip().lower()
        if self.gripper_action_convention not in {
            "dataset_zero_one",
            "simulator_minus_one_one",
        }:
            raise ValueError(
                "gripper_action_convention must be dataset_zero_one or "
                f"simulator_minus_one_one, got {gripper_action_convention!r}."
            )
        self._semantic_cache_warned = False
        if isinstance(video_augmentation, DictConfig):
            video_augmentation = OmegaConf.to_container(video_augmentation, resolve=True)
        self.video_augmentation = None
        if video_augmentation is not None and is_training_set:
            self.video_augmentation = SynchronizedVideoAugmentation(dict(video_augmentation))
            logger.info(
                "Enabled synchronized training video augmentation: %s",
                dict(video_augmentation),
            )
        elif video_augmentation is not None:
            logger.info("Ignoring video_augmentation because is_training_set=False")
        self.qwen_instruction_by_task = self._load_qwen_instruction_cache(qwen_instruction_cache_path)
        self.vision_feature_cache_dir = vision_feature_cache_dir
        if vision_feature_cache_feature_spec is None:
            self.vision_feature_cache_feature_spec = None
        elif isinstance(vision_feature_cache_feature_spec, DictConfig):
            self.vision_feature_cache_feature_spec = dict(
                OmegaConf.to_container(vision_feature_cache_feature_spec, resolve=True)
            )
        else:
            self.vision_feature_cache_feature_spec = dict(vision_feature_cache_feature_spec)
        self.vision_feature_cache = None
        if vision_feature_cache_dir is not None and str(vision_feature_cache_dir).strip():
            self.vision_feature_cache = VisionFeatureCache(
                vision_feature_cache_dir,
                dataset_length=len(self),
                expected_spec=self.vision_cache_spec(),
                expected_feature_spec=self.vision_feature_cache_feature_spec,
            )
            logger.info(
                "Using vision feature cache %s with arrays=%s",
                vision_feature_cache_dir,
                self.vision_feature_cache.keys,
            )
        required_image_free_keys = {
            "video_latents",
            "video_dino_tokens",
            "history_dino_tokens",
        }
        self.vision_cache_skips_image_decode = bool(
            vision_feature_cache_skip_image_decode
            and self.vision_feature_cache is not None
            and required_image_free_keys.issubset(self.vision_feature_cache.keys)
        )
        if vision_feature_cache_skip_image_decode and not self.vision_cache_skips_image_decode:
            available_cache_keys = (
                set(self.vision_feature_cache.keys) if self.vision_feature_cache is not None else set()
            )
            missing = sorted(
                required_image_free_keys
                - available_cache_keys
            )
            raise ValueError(
                "vision_feature_cache_skip_image_decode=True requires a complete cache containing "
                f"{sorted(required_image_free_keys)}; missing={missing}."
            )
        if self.vision_cache_skips_image_decode:
            self.lerobot_dataset._set_return_images(False)
            self.history_lerobot_dataset._set_return_images(False)
            logger.info("Complete VAE/DINO cache enabled; raw video decode and resize are disabled.")
        if self.video_augmentation is not None and self.vision_feature_cache is not None:
            raise ValueError(
                "video_augmentation cannot be combined with vision_feature_cache: cached visual "
                "features would bypass the requested image augmentation."
            )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
            self.history_lerobot_dataset.set_processor(processor)

    def _load_qwen_instruction_cache(self, cache_path: Optional[str]) -> dict[str, str]:
        if cache_path is None or str(cache_path).strip() == "":
            return {}
        path = os.path.expanduser(str(cache_path))
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing Qwen instruction cache: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("records", payload)
        if isinstance(records, dict):
            iterator = records.values()
        elif isinstance(records, list):
            iterator = records
        else:
            raise ValueError(
                f"Qwen instruction cache must contain a records dict/list, got {type(records)}."
            )

        mapping: dict[str, str] = {}
        for record in iterator:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid Qwen instruction record type: {type(record)}")
            task = str(record.get("task", "")).strip()
            if not task:
                raise ValueError(f"Qwen instruction record missing non-empty `task`: {record}")
            enhanced_task = str(record.get("enhanced_task", "")).strip()
            if not enhanced_task:
                answer = str(record.get("qwen_answer", "")).strip()
                if not answer:
                    raise ValueError(
                        f"Qwen instruction record for task={task!r} must contain "
                        "`enhanced_task` or non-empty `qwen_answer`."
                    )
                enhanced_task = f"{task} {answer}"
            mapping[task] = enhanced_task
        logger.info("Loaded %d Qwen-enhanced instructions from %s", len(mapping), path)
        return mapping

    def _apply_qwen_instruction_cache(self, task: str) -> str:
        if not self.qwen_instruction_by_task:
            return task
        enhanced = self.qwen_instruction_by_task.get(task)
        if enhanced is not None:
            return enhanced
        if self.qwen_instruction_missing == "original":
            logger.warning("Missing Qwen-enhanced instruction for task=%r; using original.", task)
            return task
        raise KeyError(
            f"Missing Qwen-enhanced instruction for task={task!r}. "
            "Regenerate the Qwen instruction cache or set qwen_instruction_missing=original."
        )
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def vision_cache_spec(self) -> dict[str, object]:
        return {
            "dataset_dirs": [str(path) for path in self.lerobot_dataset.dataset_dirs],
            "num_frames": int(self.num_frames),
            "action_video_freq_ratio": int(self.action_video_freq_ratio),
            "video_size": [int(x) for x in self.video_size],
            "video_sample_indices": [int(x) for x in self.video_sample_indices],
            "concat_multi_camera": self.concat_multi_camera,
            "is_training_set": bool(self.lerobot_dataset.is_training_set),
        }

    def _format_video_sample(self, sample, augmentation_params=None):
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.video_augmentation is not None:
            video, augmentation_params = self.video_augmentation(video, augmentation_params)
        if self.concat_multi_camera == "robotwin":
            video = compose_robotwin_cameras(video)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]
        return video, image_is_pad, augmentation_params

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        history_sample = self.history_lerobot_dataset[sample_idx]
        if self.vision_cache_skips_image_decode:
            sampled_frames = len(self.video_sample_indices)
            video = torch.zeros((3, sampled_frames, 1, 1), dtype=torch.float32)
            history_video = torch.zeros_like(video)
            image_is_pad = sample["image_is_pad"][self.video_sample_indices]
            history_image_is_pad = history_sample["image_is_pad"][self.video_sample_indices]
        else:
            video, image_is_pad, augmentation_params = self._format_video_sample(sample)
            history_video, history_image_is_pad, _ = self._format_video_sample(
                history_sample,
                augmentation_params=augmentation_params,
            )

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        history_action = history_sample["action"].clone() # [-T..-1, action_dim]
        history_state = history_sample["proprio"].clone() # [-T..0, state_dim], one more state than actions
        history_action_is_pad = history_sample["action_is_pad"].bool()
        if bool(history_action_is_pad.any().item()):
            history_action[history_action_is_pad] = 0.0
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        else:
            task = self._apply_qwen_instruction_cache(str(task))
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        semantic_context = context
        semantic_context_mask = context_mask
        semantic_prompt = instruction
        if self.semantic_text_target == "task":
            semantic_prompt = str(task)
            try:
                semantic_context, semantic_context_mask = self._get_cached_text_context(semantic_prompt)
                semantic_context[~semantic_context_mask] = 0.0
                semantic_context_mask = torch.ones_like(semantic_context_mask)
            except FileNotFoundError:
                if not self.semantic_text_fallback_to_prompt:
                    raise
                if not self._semantic_cache_warned:
                    logger.warning(
                        "Missing task-only text embedding cache for semantic alignment; "
                        "falling back to the full prompt target. Run scripts/precompute_text_embeds.py "
                        "with include_task_only_prompts=true to enable task-only alignment."
                    )
                    self._semantic_cache_warned = True
                semantic_context = context
                semantic_context_mask = context_mask
                semantic_prompt = instruction
        elif self.semantic_text_target == "none":
            semantic_prompt = ""
        
        data = {
            "sample_index": torch.as_tensor(sample_idx, dtype=torch.long),
            "video": video,
            "history_video": history_video,
            "action": action,
            "history_action": history_action,
            "history_state": history_state,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "semantic_prompt": semantic_prompt,
            "semantic_context": semantic_context,
            "semantic_context_mask": semantic_context_mask,
            "image_is_pad": image_is_pad,
            "history_image_is_pad": history_image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "history_action_is_pad": history_action_is_pad,
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.vision_feature_cache is not None:
            data.update(self.vision_feature_cache.get(sample_idx))
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            if self.strict_sample_errors:
                raise RuntimeError(f"Failed to process dataset sample idx={idx}.") from e
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
