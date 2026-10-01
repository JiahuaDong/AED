import hashlib
import json
import logging
import math
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from wam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from wam.utils.config_resolvers import register_default_resolvers
from wam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _collect_dataset_settings(data_cfg: DictConfig):
    dataset_dirs: list[str] = []
    cache_dirs: list[Path] = []
    context_lens = set()
    dataset_nodes: list[tuple[str, DictConfig]] = []

    for node_path, node in _iter_dataset_nodes(data_cfg, path="data"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue
        dataset_nodes.append((node_path, node))

        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Missing `text_embedding_cache_dir` for dataset node `{node_path}` "
                "(this node defines `dataset_dirs`)."
            )

        for ds in raw_dirs:
            ds_str = str(ds)
            if ds_str not in dataset_dirs:
                dataset_dirs.append(ds_str)

        cache_dir_path = Path(str(cache_dir)).expanduser()
        if cache_dir_path not in cache_dirs:
            cache_dirs.append(cache_dir_path)

        context_len = node.get("context_len")
        if context_len is not None:
            context_lens.add(int(context_len))

        logger.info("Discovered dataset node `%s` with %d dataset_dirs.", node_path, len(raw_dirs))

    return dataset_dirs, cache_dirs, context_lens, dataset_nodes


def _resolve_context_len(context_lens: set[int]) -> int:
    if len(context_lens) != 1:
        raise ValueError(
            f"Found multiple context_len values in data config: {sorted(context_lens)}. "
            "Please keep them consistent."
        )
    return next(iter(context_lens))


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _episode_task_key(episode: dict[str, Any], group_size: int | None = None) -> str:
    if group_size is not None:
        episode_idx = int(episode.get("episode_index", 0))
        return f"group_{episode_idx // group_size}"
    tasks = episode.get("tasks", None)
    if isinstance(tasks, list) and len(tasks) > 0:
        return " | ".join(str(task) for task in sorted(tasks))
    return "__unknown_task__"


def _select_episode_records(
    episodes: list[dict[str, Any]],
    *,
    val_set_proportion: float,
    is_training_set: bool,
    seed: int,
    episode_sample_ratio: float | None,
    episode_sample_max_per_task: int | None,
    episode_sample_min_per_task: int,
    episode_sample_group_size: int | None,
) -> list[dict[str, Any]]:
    if val_set_proportion < 1e-6:
        selected = list(episodes)
    else:
        episode_indices = list(range(len(episodes)))
        rng = np.random.default_rng(seed)
        rng.shuffle(episode_indices)
        split_idx = int(len(episodes) * (1 - val_set_proportion))
        if is_training_set:
            selected_indices = episode_indices[:split_idx]
        else:
            selected_indices = episode_indices[split_idx:]
        selected = [episodes[i] for i in selected_indices]

    has_ratio = episode_sample_ratio is not None and float(episode_sample_ratio) < 1.0
    has_max = episode_sample_max_per_task is not None
    if not has_ratio and not has_max:
        return selected

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in selected:
        groups[_episode_task_key(episode, episode_sample_group_size)].append(episode)

    rng = np.random.default_rng(seed)
    sampled: list[dict[str, Any]] = []
    for task_key in sorted(groups):
        candidates = list(groups[task_key])
        rng.shuffle(candidates)
        keep_n = len(candidates)
        if has_ratio:
            keep_n = int(math.ceil(len(candidates) * float(episode_sample_ratio)))
        if has_max:
            keep_n = min(keep_n, int(episode_sample_max_per_task))
        if episode_sample_min_per_task > 0 and not has_max:
            keep_n = max(episode_sample_min_per_task, keep_n)
        sampled.extend(candidates[: min(len(candidates), keep_n)])

    sampled.sort(key=lambda item: int(item.get("episode_index", 0)))
    return sampled


def _add_prompt(prompts: list[str], seen: set[str], prompt: str):
    if prompt not in seen:
        seen.add(prompt)
        prompts.append(prompt)


def _add_task_prompts(prompts: list[str], seen: set[str], task: str, *, include_task_only: bool):
    _add_prompt(prompts, seen, DEFAULT_PROMPT.format(task=task))
    if include_task_only:
        _add_prompt(prompts, seen, task)


def _read_unique_prompts(dataset_dirs: list[str], *, include_task_only: bool = True) -> list[str]:
    prompts: list[str] = []
    seen = set()
    total_task_rows = 0

    for ds_dir in dataset_dirs:
        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with tasks_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "task" not in record:
                        raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                    task = str(record["task"])
                    total_task_rows += 1
                    _add_task_prompts(prompts, seen, task, include_task_only=include_task_only)
            continue

        parquet_path = Path(ds_dir) / "meta" / "tasks.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing LeRobot task metadata: expected {tasks_path} or {parquet_path}."
            )
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                f"Reading LeRobot v3 task metadata requires pyarrow: {parquet_path}"
            ) from exc

        table = pq.read_table(parquet_path)
        task_column = None
        for candidate in ("task", "instruction", "__index_level_0__"):
            if candidate in table.column_names:
                task_column = candidate
                break
        if task_column is None:
            string_columns = [
                field.name
                for field in table.schema
                if str(field.type) in {"string", "large_string"}
            ]
            if len(string_columns) == 1:
                task_column = string_columns[0]
        if task_column is None:
            raise KeyError(
                f"Cannot identify task text column in {parquet_path}; columns={table.column_names}."
            )
        for task_value in table[task_column].to_pylist():
            task = str(task_value).strip()
            if not task:
                raise ValueError(f"Found an empty task in {parquet_path}:{task_column}.")
            total_task_rows += 1
            _add_task_prompts(prompts, seen, task, include_task_only=include_task_only)

    logger.info(
        "Loaded %d task rows from %d datasets, deduplicated to %d prompts.",
        total_task_rows,
        len(dataset_dirs),
        len(prompts),
    )
    return prompts


def _read_qwen_enhanced_prompts_from_dataset_nodes(
    dataset_nodes: list[tuple[str, DictConfig]],
    *,
    include_task_only: bool = True,
) -> list[str]:
    prompts: list[str] = []
    seen_prompts: set[str] = set()
    seen_paths: set[Path] = set()

    for node_path, node in dataset_nodes:
        cache_path_value = node.get("qwen_instruction_cache_path")
        if cache_path_value is None or not str(cache_path_value).strip():
            continue
        cache_path = Path(str(cache_path_value)).expanduser()
        if cache_path in seen_paths:
            continue
        seen_paths.add(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Dataset node `{node_path}` references missing qwen_instruction_cache_path: {cache_path}"
            )

        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("records", payload)
        if isinstance(records, dict):
            iterator = records.values()
        elif isinstance(records, list):
            iterator = records
        else:
            raise ValueError(
                f"Qwen instruction cache must contain a records dict/list, got {type(records)} in {cache_path}"
            )

        added = 0
        for record in iterator:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid Qwen instruction record type in {cache_path}: {type(record)}")
            enhanced_task = str(record.get("enhanced_task", "")).strip()
            if not enhanced_task:
                task = str(record.get("task", "")).strip()
                answer = str(record.get("qwen_answer", "")).strip()
                if not task or not answer:
                    raise ValueError(
                        f"Qwen record in {cache_path} must contain `enhanced_task` or `task`+`qwen_answer`: {record}"
                    )
                enhanced_task = f"{task} {answer}"
            before = len(prompts)
            _add_task_prompts(prompts, seen_prompts, enhanced_task, include_task_only=include_task_only)
            added += len(prompts) - before
        logger.info("Loaded %d Qwen-enhanced prompts from %s.", added, cache_path)

    return prompts


def _read_unique_prompts_from_dataset_nodes(
    dataset_nodes: list[tuple[str, DictConfig]],
    *,
    seed: int,
    include_task_only: bool = True,
) -> list[str]:
    prompts: list[str] = []
    seen = set()
    total_episode_rows = 0
    total_selected_episode_rows = 0
    any_subset = False

    for node_idx, (node_path, node) in enumerate(dataset_nodes):
        ratio = _to_optional_float(node.get("episode_sample_ratio"))
        max_per_task = _to_optional_int(node.get("episode_sample_max_per_task"))
        min_per_task = int(node.get("episode_sample_min_per_task", 1))
        group_size = _to_optional_int(node.get("episode_sample_group_size"))
        has_subset = (ratio is not None and ratio < 1.0) or max_per_task is not None
        if not has_subset:
            continue
        any_subset = True

        val_set_proportion = float(node.get("val_set_proportion", 0.0))
        is_training_set = _to_bool(node.get("is_training_set", False))
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue

        for ds_idx, ds_dir in enumerate(raw_dirs):
            episodes_path = Path(str(ds_dir)) / "meta" / "episodes.jsonl"
            if not episodes_path.exists():
                raise FileNotFoundError(f"Missing episodes file: {episodes_path}")

            episodes: list[dict[str, Any]] = []
            with episodes_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "tasks" not in record:
                        raise KeyError(f"Missing `tasks` field at {episodes_path}:{line_idx}")
                    episodes.append(record)

            selected = _select_episode_records(
                episodes,
                val_set_proportion=val_set_proportion,
                is_training_set=is_training_set,
                seed=seed + ds_idx,
                episode_sample_ratio=ratio,
                episode_sample_max_per_task=max_per_task,
                episode_sample_min_per_task=min_per_task,
                episode_sample_group_size=group_size,
            )
            total_episode_rows += len(episodes)
            total_selected_episode_rows += len(selected)
            logger.info(
                "Dataset node `%s` selected %d/%d episodes from %s for text precompute "
                "(ratio=%s, max_per_task=%s, min_per_task=%d, group_size=%s).",
                node_path,
                len(selected),
                len(episodes),
                ds_dir,
                ratio,
                max_per_task,
                min_per_task,
                group_size,
            )

            for episode in selected:
                for task in episode.get("tasks", []):
                    _add_task_prompts(prompts, seen, str(task), include_task_only=include_task_only)

    if not any_subset:
        return []

    logger.info(
        "Loaded %d selected episode rows from %d total episode rows, deduplicated to %d prompts.",
        total_selected_episode_rows,
        total_episode_rows,
        len(prompts),
    )
    return prompts


def _get_override_prompts(override_instruction: Any, *, include_task_only: bool = True) -> list[str] | None:
    if override_instruction is None:
        return None
    task = str(override_instruction).strip()
    if task == "":
        return None
    prompts = [DEFAULT_PROMPT.format(task=task)]
    if include_task_only:
        prompts.append(task)
    return prompts


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d scripts/precompute_text_embeds.py",
            torch.cuda.device_count(),
        )

    overwrite = _to_bool(cfg.get("overwrite", True))
    model_cfg = cfg.model
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")

    dataset_dirs, cache_dirs, context_lens, dataset_nodes = _collect_dataset_settings(cfg.data)
    if not cache_dirs:
        raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")

    context_len = _resolve_context_len(context_lens)
    include_task_only = _to_bool(cfg.get("include_task_only_prompts", True))
    override_prompts = _get_override_prompts(
        cfg.get("override_instruction"),
        include_task_only=include_task_only,
    )
    if override_prompts is not None:
        prompts = override_prompts
        logger.info(
            "Using override_instruction; skipping dataset scan and encoding exactly %d prompt(s).",
            len(prompts),
        )
    else:
        if not dataset_dirs:
            raise ValueError("No `dataset_dirs` found under `cfg.data`.")
        prompts = _read_qwen_enhanced_prompts_from_dataset_nodes(
            dataset_nodes,
            include_task_only=include_task_only,
        )
        if prompts:
            logger.info("Using %d Qwen-enhanced prompts from dataset nodes.", len(prompts))
        else:
            prompts = _read_unique_prompts_from_dataset_nodes(
                dataset_nodes,
                seed=int(cfg.get("seed", 42)),
                include_task_only=include_task_only,
            )
        if not prompts:
            prompts = _read_unique_prompts(dataset_dirs, include_task_only=include_task_only)
    if not prompts:
        logger.warning("No prompts found from tasks.jsonl; nothing to do.")
        return

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    enc_id = _model_id_to_enc_id(model_id)

    logger.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s device=%s dtype=%s context_len=%d overwrite=%s",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    stats = {
        str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
        for cache_dir in cache_dirs
    }

    prompts = prompts[rank::world_size] if is_distributed else prompts

    if not overwrite:
        fully_cached_local = 0
        prompts_to_encode: list[str] = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            fully_cached = True
            for cache_dir in cache_dirs:
                cache_path = cache_dir / filename
                if not cache_path.exists():
                    fully_cached = False
                    break
            if fully_cached:
                fully_cached_local += 1
                for cache_dir in cache_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                prompts_to_encode.append(prompt)

        prompts = prompts_to_encode

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompts)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    logger.info("Writing caches to %d directories.", len(cache_dirs))
    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    over_length_prompts = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_prompts += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {
                        "context": context_i,
                        "mask": mask_i,
                    }

                    for cache_dir in cache_dirs:
                        cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                        key = str(cache_dir)
                        if cache_path.exists() and not overwrite:
                            stats[key]["skip"] += 1
                            continue

                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1

                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [
                [stats[str(cache_dir)]["new"], stats[str(cache_dir)]["overwrite"], stats[str(cache_dir)]["skip"]]
                for cache_dir in cache_dirs
            ],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            prompts_encoded_global,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
