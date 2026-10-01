"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import time
import pathlib
import re
from experiments.libero.prompt_protocol import get_training_prompt_entry

import imageio
from PIL import Image, ImageDraw
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import libero.libero.envs.bddl_utils as BDDLUtils
from wam.utils.video_io import save_mp4

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256  # simulator render size (not model input size)


_LIBERO_PLUS_NUMBERED_SUFFIX = re.compile(
    r"(?:_(?:table|tb|light|add)_\d+(?:_|$)|_level\d+_sample\d+(?:_|$))",
)


def is_libero_plus_variant(task_name: str) -> bool:
    """Return whether a task name carries LIBERO-Plus perturbation metadata."""
    name = str(task_name)
    return (
        "_language_" in name
        or ("_view_" in name and "_initstate_" in name)
        or "_noise_" in name
        or _LIBERO_PLUS_NUMBERED_SUFFIX.search(name) is not None
    )


def resolve_libero_prompt_source(task, prompt_source: str, training_instruction_map=None) -> str:
    """Resolve ``auto`` without changing standard-LIBERO prompt wording."""
    if prompt_source == "training_nonlanguage":
        entry = get_training_prompt_entry(task, training_instruction_map)
        return "bddl_language" if entry["category"] == "Language Instructions" else "training_instruction"
    if prompt_source not in ("auto", "task.language", "bddl_language"):
        raise ValueError(
            "prompt_source must be 'auto', 'task.language', or "
            f"'bddl_language', got {prompt_source!r}"
        )
    if prompt_source != "auto":
        return prompt_source
    return "bddl_language" if is_libero_plus_variant(task.name) else "task.language"


def _resolve_bddl_for_language(task_bddl_file: pathlib.Path) -> pathlib.Path:
    """Map LIBERO-Plus shared-state variants to the BDDL that owns language."""
    path = pathlib.Path(task_bddl_file)
    filename = path.name
    if "_view_" in filename and "_initstate_" in filename:
        return path.with_name(filename.split("_view_", 1)[0] + ".bddl")
    return path


def _read_bddl_language(task_bddl_file: pathlib.Path) -> str:
    language_bddl = _resolve_bddl_for_language(task_bddl_file)
    if not language_bddl.is_file():
        raise FileNotFoundError(
            f"Cannot resolve LIBERO prompt because BDDL file is missing: {language_bddl}"
        )
    parsed = BDDLUtils.robosuite_parse_problem(str(language_bddl))
    language = parsed.get("language_instruction")
    if isinstance(language, list):
        language = " ".join(str(token) for token in language)
    language = str(language or "").strip()
    if not language:
        raise ValueError(f"BDDL file has no non-empty :language field: {language_bddl}")
    return language


def get_libero_env(task, resolution, seed, env_num=1, prompt_source="auto", training_instruction_map=None):
    """Initialize LIBERO and return the environment plus policy instruction.

    ``auto`` uses filename-derived instructions for standard LIBERO, whose
    training wording can differ from the BDDL annotation, and the BDDL
    ``:language`` field for LIBERO-Plus variants, so perturbation metadata
    does not end up in the prompt.
    """
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    resolved_prompt_source = resolve_libero_prompt_source(task, str(prompt_source), training_instruction_map)
    if resolved_prompt_source == "task.language":
        task_description = str(task.language)
    elif resolved_prompt_source == "training_instruction":
        task_description = get_training_prompt_entry(task, training_instruction_map)["instruction"]
    else:
        task_description = _read_bddl_language(task_bddl_file)
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if env_num > 1:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    # [yc] wrist image
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    return {
        "image": img,
        "wrist_image": wrist_img
    }

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None, fps=24):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            image = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                image.append(np.array(pil_img))
            frame = np.concatenate(image, axis=1)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_prediction_video(
    rollout_dir,
    gt_frames,
    pred_frames,
    idx,
    replan_idx,
    success,
    task_description,
    log_file=None,
    fps=8,
):
    """Saves an MP4 comparison of ground-truth and predicted future frames for one replanning clip."""
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames <= 0:
        raise ValueError("Cannot save prediction video with empty GT/pred frame lists.")

    stitched_frames = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        if isinstance(gt_frame, dict):
            gt_images = []
            for value in gt_frame.values():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                gt_images.append(value_array)
            gt_image = np.concatenate(gt_images, axis=1)
        elif isinstance(gt_frame, Image.Image):
            gt_image = np.array(gt_frame.convert("RGB"))
        else:
            gt_image = np.array(gt_frame)

        if isinstance(pred_frame, Image.Image):
            pred_image = np.array(pred_frame.convert("RGB"))
        else:
            pred_image = np.array(pred_frame)

        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_pil = Image.fromarray(gt_image)
        ImageDraw.Draw(gt_pil).text((10, 10), "gt", fill=(255, 255, 255))
        pred_pil = Image.fromarray(pred_image)
        ImageDraw.Draw(pred_pil).text((10, 10), "pred", fill=(255, 255, 255))
        stitched_frames.append(
            Image.fromarray(np.concatenate([np.array(pred_pil), np.array(gt_pil)], axis=0))
        )

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    try:
        replan_tag = f"{int(replan_idx):04d}"
    except (TypeError, ValueError):
        replan_tag = str(replan_idx)
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--replan={replan_tag}--gt-pred.mp4"
    )
    save_mp4(stitched_frames, mp4_path, fps=fps)
    print(f"Saved predicted future comparison MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved predicted future comparison MP4 at path {mp4_path}\n")
    return mp4_path

def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = (v > 0.5)
    return np.asarray(bin_val, dtype=np.float32)


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
