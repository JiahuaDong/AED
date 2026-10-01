from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq


VALID_EPISODE_SAMPLE_GROUP_KEYS = {
    "legacy",
    "episode_group",
    "episode_tasks",
    "instruction",
    "first_instruction",
}

_FRAME_INSTRUCTION_CACHE: dict[tuple[str, int, bool], str] = {}


def resolve_episode_sample_max(
    episode_sample_max_per_task: Optional[int],
    episode_sample_max_per_instruction: Optional[int],
) -> Optional[int]:
    if episode_sample_max_per_instruction is None:
        return None if episode_sample_max_per_task is None else int(episode_sample_max_per_task)
    if episode_sample_max_per_task is not None and int(episode_sample_max_per_task) != int(episode_sample_max_per_instruction):
        raise ValueError(
            "`episode_sample_max_per_task` and `episode_sample_max_per_instruction` "
            "are aliases; set only one or use the same value."
        )
    return int(episode_sample_max_per_instruction)


def normalize_episode_sample_group_key(value: Optional[str]) -> str:
    if value is None:
        return "legacy"
    key = str(value).strip().lower()
    aliases = {
        "task": "episode_tasks",
        "tasks": "episode_tasks",
        "episode_task": "episode_tasks",
        "episode-task": "episode_tasks",
        "instruction": "instruction",
        "instructions": "instruction",
        "frame_task": "instruction",
        "frame-task": "instruction",
        "frame_instruction": "instruction",
        "frame-instruction": "instruction",
        "first_task": "first_instruction",
        "first-task": "first_instruction",
        "first_instruction": "first_instruction",
        "first-instruction": "first_instruction",
        "group": "episode_group",
        "episode_group": "episode_group",
        "episode-group": "episode_group",
    }
    key = aliases.get(key, key)
    if key not in VALID_EPISODE_SAMPLE_GROUP_KEYS:
        valid = ", ".join(sorted(VALID_EPISODE_SAMPLE_GROUP_KEYS))
        raise ValueError(f"`episode_sample_group_key` must be one of {{{valid}}}, got {value!r}.")
    return key


def _episode_record(meta: Any, episode_idx: int) -> dict[str, Any]:
    return meta.episodes.get(episode_idx, meta.episodes.get(str(episode_idx), {}))


def _episode_tasks_key(meta: Any, episode_idx: int, *, first_only: bool = False) -> str:
    episode = _episode_record(meta, episode_idx)
    tasks = episode.get("tasks", None)
    if isinstance(tasks, list) and len(tasks) > 0:
        if first_only:
            return str(tasks[0])
        return " | ".join(str(task) for task in sorted(tasks))
    return "__unknown_task__"


def _frame_instruction_key(meta: Any, episode_idx: int, *, first_only: bool = False) -> str:
    cache_key = (str(meta.root), int(episode_idx), bool(first_only))
    if cache_key in _FRAME_INSTRUCTION_CACHE:
        return _FRAME_INSTRUCTION_CACHE[cache_key]
    try:
        data_path = meta.root / meta.get_data_file_path(episode_idx)
        task_indices = pq.read_table(str(data_path), columns=["task_index"])["task_index"].to_numpy()
    except Exception:
        key = _episode_tasks_key(meta, episode_idx, first_only=first_only)
        _FRAME_INSTRUCTION_CACHE[cache_key] = key
        return key

    unique_task_indices = []
    seen = set()
    for value in np.asarray(task_indices).reshape(-1):
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        unique_task_indices.append(idx)
        if first_only:
            break

    tasks = [str(meta.tasks.get(idx, idx)) for idx in unique_task_indices]
    if tasks:
        key = tasks[0] if first_only else " | ".join(sorted(tasks))
    else:
        key = _episode_tasks_key(meta, episode_idx, first_only=first_only)
    _FRAME_INSTRUCTION_CACHE[cache_key] = key
    return key


def episode_sample_group_key(
    meta: Any,
    episode_idx: int,
    *,
    group_key: str,
    group_size: Optional[int],
) -> str:
    if group_key == "legacy":
        if group_size is not None:
            return f"group_{episode_idx // group_size}"
        return _episode_tasks_key(meta, episode_idx)
    if group_key == "episode_group":
        if group_size is None:
            raise ValueError("`episode_sample_group_size` is required when group_key='episode_group'.")
        return f"group_{episode_idx // group_size}"
    if group_key == "episode_tasks":
        return _episode_tasks_key(meta, episode_idx)
    if group_key == "instruction":
        return _frame_instruction_key(meta, episode_idx)
    if group_key == "first_instruction":
        return _frame_instruction_key(meta, episode_idx, first_only=True)
    raise AssertionError(f"Unhandled episode sample group key: {group_key}")
