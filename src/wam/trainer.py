import logging
import json
import inspect
import os
import re
import shutil
import tempfile
from math import ceil
from pathlib import Path
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.dataloader_persistent_workers = bool(
            cfg.get("dataloader_persistent_workers", self.num_workers > 0)
        )
        prefetch_factor = cfg.get("dataloader_prefetch_factor", 2)
        self.dataloader_prefetch_factor = (
            None if prefetch_factor is None else int(prefetch_factor)
        )
        self.dataloader_drop_last = bool(cfg.get("dataloader_drop_last", False))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_at_steps = self._normalize_save_at_steps(cfg.get("save_at_steps", []))
        self.save_final = bool(cfg.get("save_final", True))
        self.save_training_state = bool(cfg.get("save_training_state", True))
        max_training_state_checkpoints = cfg.get("max_training_state_checkpoints")
        self.max_training_state_checkpoints = (
            None
            if max_training_state_checkpoints is None
            else int(max_training_state_checkpoints)
        )
        if (
            self.max_training_state_checkpoints is not None
            and self.max_training_state_checkpoints < 1
        ):
            raise ValueError(
                "max_training_state_checkpoints must be null or a positive integer, "
                f"got {self.max_training_state_checkpoints}."
            )
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.reset_scheduler_on_state_resume = bool(
            cfg.get("reset_scheduler_on_state_resume", False)
        )
        resume_scheduler_steps = cfg.get("resume_scheduler_steps")
        self.resume_scheduler_steps = (
            None if resume_scheduler_steps is None else int(resume_scheduler_steps)
        )
        self.resume_scheduler_warmup_fraction = float(
            cfg.get("resume_scheduler_warmup_fraction", 0.05)
        )
        if self.resume_scheduler_steps is not None and self.resume_scheduler_steps < 1:
            raise ValueError(
                "resume_scheduler_steps must be null or a positive integer, "
                f"got {self.resume_scheduler_steps}."
            )
        if not 0.0 <= self.resume_scheduler_warmup_fraction < 1.0:
            raise ValueError(
                "resume_scheduler_warmup_fraction must be in [0, 1), "
                f"got {self.resume_scheduler_warmup_fraction}."
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        mlflow_cfg = cfg.get("mlflow")
        self.mlflow_enabled = bool(
            mlflow_cfg is not None and mlflow_cfg.get("enabled", False)
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self._backward_includes_deepspeed_step = (
            str(self.accelerator.distributed_type).upper().endswith("DEEPSPEED")
        )
        self._configure_deepspeed_gradient_clipping()
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        if self._backward_includes_deepspeed_step:
            logger.info(
                "DeepSpeed timing semantics: accelerator.backward() includes gradient reduction, "
                "gradient clipping, optimizer step, and zero_grad at accumulation boundaries."
            )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        logger.info(
            "DataLoader config: num_workers=%d persistent_workers=%s prefetch_factor=%s pin_memory=%s drop_last=%s",
            self.num_workers,
            self.dataloader_persistent_workers if self.num_workers > 0 else False,
            self.dataloader_prefetch_factor if self.num_workers > 0 else None,
            torch.cuda.is_available(),
            self.dataloader_drop_last,
        )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        self._weight_checkpoint_loaded_before_prepare = False
        self._load_weight_checkpoint_before_prepare()

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = self._collect_trainable_parameters(self.model)
        trainable_param_count = sum(param.numel() for param in trainable_params)
        logger.info(
            "Optimizer trainable parameters: tensors=%d parameters=%d",
            len(trainable_params),
            trainable_param_count,
        )
        if not trainable_params:
            raise RuntimeError("No trainable parameters were collected for optimizer initialization.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        profile_steps_default = os.environ.get("WAM_PROFILE_STEPS", "0")
        self.trainer_profile_steps = int(
            os.environ.get("WAM_TRAINER_PROFILE_STEPS", profile_steps_default)
        )
        self.rank_profile_steps = int(os.environ.get("WAM_RANK_PROFILE_STEPS", "0"))
        self.rank_profile_dir = os.environ.get(
            "WAM_RANK_PROFILE_DIR",
            os.path.join(self.output_dir, "rank_profile"),
        )
        self.rank_profile_path = (
            os.path.join(self.rank_profile_dir, f"rank{self.accelerator.process_index}.jsonl")
            if self.rank_profile_steps > 0
            else None
        )

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        if self.rank_profile_path is not None:
            ensure_dir(self.rank_profile_dir)
            Path(self.rank_profile_path).write_text("", encoding="utf-8")
        logger.info(
            "Timing profile config: trainer_profile_steps=%d rank_profile_steps=%d rank_profile_path=%s",
            self.trainer_profile_steps,
            self.rank_profile_steps,
            self.rank_profile_path,
        )

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self.mlflow_run = None
        self.mlflow_module = None
        self.mlflow_synchronous = True
        self._initialize_trackers_and_resume()

    def _initialize_trackers_and_resume(self) -> None:
        try:
            self._init_wandb()
            self._init_mlflow()
            self._resume_or_load_checkpoint()

            val_size = (
                len(self.val_dataset)
                if self.val_dataset is not None
                else len(self.train_dataset)
            )
            logger.info(
                "Train/val dataset size: %d/%d", len(self.train_dataset), val_size
            )
        except BaseException:
            try:
                self._finish_trackers(exit_code=1)
            except BaseException:
                logger.exception("Failed to finalize experiment trackers after initialization error.")
            raise

    def _write_rank_profile(self, payload: dict):
        if self.rank_profile_path is None:
            return
        payload = dict(payload)
        payload.setdefault("rank", int(self.accelerator.process_index))
        payload.setdefault("local_rank", int(self.accelerator.local_process_index))
        payload.setdefault("device", str(self.accelerator.device))
        payload.setdefault("time", time.time())
        with open(self.rank_profile_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _configure_deepspeed_gradient_clipping(self) -> bool:
        """Make the trainer's max_grad_norm effective before DeepSpeed builds its engine."""
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            return False

        deepspeed_config = plugin.deepspeed_config
        configured_value = deepspeed_config.get("gradient_clipping")
        if configured_value not in (None, "auto"):
            configured_value = float(configured_value)
            if not np.isclose(configured_value, self.max_grad_norm):
                raise ValueError(
                    "DeepSpeed `gradient_clipping` conflicts with trainer `max_grad_norm`: "
                    f"{configured_value} != {self.max_grad_norm}."
                )

        deepspeed_config["gradient_clipping"] = self.max_grad_norm
        logger.info(
            "Configured DeepSpeed gradient_clipping=%.4f before accelerator.prepare().",
            self.max_grad_norm,
        )
        return True

    def _effective_global_batch_size(self) -> int:
        return (
            self.batch_size
            * int(self.accelerator.num_processes)
            * self.gradient_accumulation_steps
        )

    def _samples_per_second(self, steps_per_second: float) -> float:
        return float(steps_per_second) * self._effective_global_batch_size()

    def _backward_includes_optimizer_step(self, sync_gradients: bool) -> bool:
        return bool(self._backward_includes_deepspeed_step and sync_gradients)

    @torch.no_grad()
    def _gather_scalar_metrics(self, metrics: dict[str, object], *, device: torch.device) -> dict[str, float]:
        """Average scalar metrics across ranks with one collective."""
        names = list(metrics)
        if not names:
            return {}
        local_values = []
        for name in names:
            value = metrics[name]
            if torch.is_tensor(value):
                scalar = value.detach().to(device=device, dtype=torch.float32).reshape(-1).mean()
            else:
                scalar = torch.as_tensor(float(value), device=device, dtype=torch.float32)
            local_values.append(scalar)
        local_tensor = torch.stack(local_values)
        gathered = self.accelerator.gather(local_tensor)
        expected = self.accelerator.num_processes * len(names)
        if gathered.numel() != expected:
            raise RuntimeError(
                "Unexpected gathered metric shape: "
                f"got {tuple(gathered.shape)} ({gathered.numel()} values), expected {expected}."
            )
        global_values = gathered.reshape(self.accelerator.num_processes, len(names)).mean(dim=0)
        return dict(zip(names, global_values.cpu().tolist()))

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            id=None if self.cfg.wandb.get("id") in (None, "null", "") else str(self.cfg.wandb.id),
            resume=str(self.cfg.wandb.get("resume", "allow")),
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    @staticmethod
    def _is_sensitive_mlflow_key(path: str) -> bool:
        segments = re.split(r"[./]+", path.lower())
        exact_names = {
            "password",
            "passwd",
            "secret",
            "secret_key",
            "api_key",
            "access_token",
            "auth_token",
            "private_key",
            "client_secret",
            "token",
            "hf_token",
            "bearer_token",
            "aws_secret_access_key",
            "credential",
            "credentials",
        }
        sensitive_suffixes = (
            "_password",
            "_passwd",
            "_secret",
            "_secret_key",
            "_api_key",
            "_access_token",
            "_auth_token",
            "_private_key",
            "_client_secret",
        )
        return any(
            segment in exact_names or segment.endswith(sensitive_suffixes)
            for segment in segments
        )

    @classmethod
    def _redact_mlflow_uri(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            if not parsed.scheme or not parsed.netloc:
                return value
            host = parsed.hostname or ""
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            netloc = (
                f"<redacted>@{host}"
                if parsed.username is not None or parsed.password is not None
                else host
            )
            query = urlencode(
                [
                    (
                        key,
                        "<redacted>"
                        if cls._is_sensitive_mlflow_key(key)
                        else query_value,
                    )
                    for key, query_value in parse_qsl(
                        parsed.query, keep_blank_values=True
                    )
                ]
            )
            return urlunsplit(
                (parsed.scheme, netloc, parsed.path, query, parsed.fragment)
            )
        except ValueError:
            return "<redacted-invalid-uri>"

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @classmethod
    def _redact_mlflow_config(cls, payload):
        def visit(prefix: str, value):
            if prefix and cls._is_sensitive_mlflow_key(prefix):
                return "<redacted>"
            if (
                isinstance(value, str)
                and re.split(r"[./]+", prefix.lower())[-1:] == ["tracking_uri"]
            ):
                return cls._redact_mlflow_uri(value)
            if isinstance(value, dict):
                return {
                    key: visit(f"{prefix}.{key}" if prefix else str(key), child)
                    for key, child in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [
                    visit(f"{prefix}.{index}" if prefix else str(index), child)
                    for index, child in enumerate(value)
                ]
            return value

        return visit("", payload)

    @classmethod
    def _flatten_mlflow_params(cls, payload: dict) -> dict[str, str]:
        params: dict[str, str] = {}

        def visit(prefix: str, value) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    child_prefix = f"{prefix}.{key}" if prefix else str(key)
                    visit(child_prefix, child)
                return

            if isinstance(value, (list, tuple)):
                text = json.dumps(
                    cls._redact_mlflow_config(value),
                    sort_keys=True,
                    default=str,
                )
            elif value is None:
                text = "null"
            else:
                text = str(value)

            if cls._is_sensitive_mlflow_key(prefix):
                text = "<redacted>"
            if len(prefix) > 250:
                logger.warning("Skipping overlong MLflow parameter key: %s", prefix)
                return
            # The complete resolved config is also logged as an artifact. Keep
            # individual parameter values compact for broad server compatibility.
            params[prefix] = text if len(text) <= 5000 else text[:4997] + "..."

        visit("", payload)
        return params

    def _init_mlflow(self) -> None:
        if not self.mlflow_enabled or not self.accelerator.is_main_process:
            return
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLflow logging is enabled in config (`mlflow.enabled=true`) "
                "but mlflow-skinny is not installed."
            ) from e

        mlflow_cfg = self.cfg.mlflow
        tracking_uri = str(mlflow_cfg.get("tracking_uri", "")).strip()
        experiment_name = str(mlflow_cfg.get("experiment_name", "")).strip()
        if not tracking_uri:
            raise ValueError("`mlflow.tracking_uri` must be non-empty when MLflow is enabled.")
        if not experiment_name:
            raise ValueError(
                "`mlflow.experiment_name` must be non-empty when MLflow is enabled."
            )

        run_id_value = mlflow_cfg.get("run_id")
        run_id = (
            None
            if run_id_value in (None, "null", "")
            else str(run_id_value).strip()
        )
        run_name_value = mlflow_cfg.get("run_name")
        run_name = (
            None
            if run_name_value in (None, "null", "")
            else str(run_name_value)
        )
        self.mlflow_synchronous = bool(mlflow_cfg.get("synchronous", True))
        if not self.mlflow_synchronous:
            raise ValueError(
                "WAM requires `mlflow.synchronous=true` so tracking "
                "errors surface immediately."
            )

        configured_tags = OmegaConf.to_container(
            mlflow_cfg.get("tags", {}), resolve=True
        )
        tags = {
            str(key): (
                "<redacted>"
                if self._is_sensitive_mlflow_key(str(key))
                else str(value)
            )
            for key, value in (configured_tags or {}).items()
        }
        tags.setdefault("wam.output_dir", self.output_dir)
        wandb_id = self.cfg.wandb.get("id")
        if wandb_id not in (None, "null", ""):
            tags.setdefault("wam.wandb_id", str(wandb_id))

        if tracking_uri.startswith(("http://127.0.0.1:", "http://localhost:")):
            # Requests honors proxy variables even for loopback unless NO_PROXY
            # is configured. Keep local tracking traffic off external proxies.
            for env_name in ("NO_PROXY", "no_proxy"):
                entries = [
                    item.strip()
                    for item in os.environ.get(env_name, "").split(",")
                    if item.strip()
                ]
                for host in ("127.0.0.1", "localhost"):
                    if host not in entries:
                        entries.append(host)
                os.environ[env_name] = ",".join(entries)

        mlflow.set_tracking_uri(tracking_uri)
        existing_experiment = None
        if run_id is not None:
            existing_run = mlflow.get_run(run_id)
            existing_experiment = mlflow.get_experiment(
                existing_run.info.experiment_id
            )
            if existing_experiment.name != experiment_name:
                raise ValueError(
                    "Explicit MLflow run belongs to a different experiment: "
                    f"run_id={run_id} configured={experiment_name!r} "
                    f"actual={existing_experiment.name!r}."
                )
        if run_id is None:
            mlflow.set_experiment(experiment_name)
        self.mlflow_run = mlflow.start_run(
            run_id=run_id,
            run_name=None if run_id is not None else run_name,
            tags=tags,
        )
        self.mlflow_module = mlflow

        active_run_id = str(self.mlflow_run.info.run_id)
        active_experiment_id = str(self.mlflow_run.info.experiment_id)
        active_experiment_name = (
            str(existing_experiment.name)
            if existing_experiment is not None
            else experiment_name
        )
        active_run_name = (
            str(self.mlflow_run.info.run_name)
            if self.mlflow_run.info.run_name is not None
            else None
        )
        safe_tracking_uri = self._redact_mlflow_uri(tracking_uri)
        run_url = ""
        if tracking_uri.startswith(("http://", "https://")):
            run_url = (
                f"{safe_tracking_uri.rstrip('/')}/#/experiments/"
                f"{active_experiment_id}/runs/{active_run_id}"
            )
        run_metadata = {
            "tracking_uri": safe_tracking_uri,
            "experiment_name": active_experiment_name,
            "experiment_id": active_experiment_id,
            "run_id": active_run_id,
            "run_name": active_run_name,
            "run_url": run_url,
            "artifact_uri": str(self.mlflow_run.info.artifact_uri),
            "resumed": run_id is not None,
        }
        self._atomic_write_text(
            Path(self.output_dir, "mlflow_run_id.txt"),
            active_run_id + "\n",
        )
        self._atomic_write_text(
            Path(self.output_dir, "mlflow_run.json"),
            json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        )

        if run_id is None:
            resolved_cfg = OmegaConf.to_container(self.cfg, resolve=True)
            redacted_cfg = self._redact_mlflow_config(resolved_cfg)
            params = self._flatten_mlflow_params(redacted_cfg)
            items = list(params.items())
            for start in range(0, len(items), 100):
                mlflow.log_params(
                    dict(items[start : start + 100]),
                    synchronous=True,
                )
            if bool(mlflow_cfg.get("log_config_artifact", True)):
                config_path = Path(self.output_dir, "config.yaml")
                if not config_path.is_file():
                    raise FileNotFoundError(
                        f"MLflow config artifact is missing: {config_path}"
                    )
                redacted_config_path = Path(
                    self.output_dir, "resolved_config.redacted.yaml"
                )
                redacted_config_path.write_text(
                    OmegaConf.to_yaml(OmegaConf.create(redacted_cfg), resolve=True),
                    encoding="utf-8",
                )
                mlflow.log_artifact(
                    str(redacted_config_path), artifact_path="repro"
                )

        logger.info(
            "Initialized MLflow run: tracking_uri=%s experiment=%s run_id=%s "
            "name=%s resumed=%s synchronous=%s",
            safe_tracking_uri,
            active_experiment_name,
            active_run_id,
            active_run_name,
            run_id is not None,
            self.mlflow_synchronous,
        )

    def _mlflow_log(self, payload: dict) -> None:
        if self.mlflow_run is None:
            return
        metrics: dict[str, float] = {}
        for key, value in payload.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(
                        f"MLflow metric `{key}` must be scalar, got shape {tuple(value.shape)}."
                    )
                value = value.detach().float().item()
            if isinstance(value, (bool, int, float, np.number)):
                metrics[str(key)] = float(value)
                continue
            raise TypeError(
                f"MLflow metric `{key}` must be numeric, got {type(value).__name__}."
            )
        self.mlflow_module.log_metrics(
            metrics,
            step=int(self.global_step),
            synchronous=self.mlflow_synchronous,
        )

    def _log_tracking_metrics(self, payload: dict) -> None:
        self._wandb_log(payload)
        self._mlflow_log(payload)

    def _tracking_runtime_metrics(
        self,
        *,
        steps_per_sec: float,
        data_time: float,
        forward_time: float,
        backward_time: float,
        optimizer_time: float,
        iteration_time: float,
        optimizer_step_skipped: bool,
    ) -> dict[str, float | int]:
        elapsed = max(time.perf_counter() - self.run_start_time, 0.0)
        remaining_steps = max(int(self.max_steps) - self.global_step, 0)
        eta_seconds = remaining_steps / max(float(steps_per_sec), 1e-9)
        metrics: dict[str, float | int] = {
            "trainer/global_step": int(self.global_step),
            "trainer/epoch": int(self.epoch),
            "trainer/batch_in_epoch": int(self.batch_in_epoch),
            "trainer/progress_fraction": self.global_step / max(int(self.max_steps), 1),
            "trainer/optimizer_step_skipped": int(optimizer_step_skipped),
            "performance/elapsed_sec": elapsed,
            "performance/eta_sec": eta_seconds,
            "performance/data_time_sec": float(data_time),
            "performance/forward_time_sec": float(forward_time),
            "performance/backward_time_sec": float(backward_time),
            "performance/optimizer_wrapper_time_sec": float(optimizer_time),
            "performance/iteration_time_sec": float(iteration_time),
            "performance/backward_includes_deepspeed_step": int(
                self._backward_includes_optimizer_step(sync_gradients=True)
            ),
        }
        if torch.cuda.is_available():
            device = self.accelerator.device
            bytes_per_gib = float(1024**3)
            metrics.update(
                {
                    "memory/cuda_allocated_gib": torch.cuda.memory_allocated(device) / bytes_per_gib,
                    "memory/cuda_reserved_gib": torch.cuda.memory_reserved(device) / bytes_per_gib,
                    "memory/cuda_peak_allocated_gib": (
                        torch.cuda.max_memory_allocated(device) / bytes_per_gib
                    ),
                }
            )
        return metrics

    def _wandb_record_checkpoint(self, checkpoint_info: dict) -> None:
        if self.wandb_run is None:
            return
        weights_path = checkpoint_info.get("weights_path")
        state_path = checkpoint_info.get("state_path")
        self.wandb_run.summary["checkpoint/latest_step"] = int(self.global_step)
        if weights_path is not None:
            self.wandb_run.summary["checkpoint/latest_weights_path"] = str(weights_path)
        if state_path is not None:
            self.wandb_run.summary["checkpoint/latest_state_path"] = str(state_path)
        self._wandb_log(
            {
                "checkpoint/saved": 1,
                "checkpoint/step": int(self.global_step),
            }
        )

    def _mlflow_record_checkpoint(self, checkpoint_info: dict) -> None:
        if self.mlflow_run is None:
            return
        weights_path = checkpoint_info.get("weights_path")
        state_path = checkpoint_info.get("state_path")
        tags = {"checkpoint.latest_step": str(int(self.global_step))}
        if weights_path is not None:
            tags["checkpoint.latest_weights_path"] = str(weights_path)
        if state_path is not None:
            tags["checkpoint.latest_state_path"] = str(state_path)
        self.mlflow_module.set_tags(tags, synchronous=True)
        self._mlflow_log(
            {
                "checkpoint/saved": 1,
                "checkpoint/step": int(self.global_step),
            }
        )

    def _record_checkpoint(self, checkpoint_info: dict) -> None:
        self._wandb_record_checkpoint(checkpoint_info)
        self._mlflow_record_checkpoint(checkpoint_info)

    def _finish_wandb(self, *, exit_code: int = 0):
        if self.wandb_run is None:
            return
        self.wandb_run.finish(exit_code=int(exit_code))
        self.wandb_run = None

    def _finish_mlflow(self, *, exit_code: int = 0) -> None:
        if self.mlflow_run is None:
            return
        status = "FINISHED" if int(exit_code) == 0 else "FAILED"
        errors: list[BaseException] = []
        try:
            self.mlflow_module.set_tags(
                {
                    "run.exit_code": str(int(exit_code)),
                    "run.status": status,
                },
                synchronous=True,
            )
        except BaseException as exc:
            logger.exception("Failed to set final MLflow run tags.")
            errors.append(exc)
        try:
            self.mlflow_module.end_run(status=status)
        except BaseException as exc:
            logger.exception("Failed to end MLflow run with status %s.", status)
            errors.append(exc)
        finally:
            self.mlflow_run = None
            self.mlflow_module = None
        if errors:
            raise errors[0]

    def _finish_trackers(self, *, exit_code: int = 0) -> None:
        errors: list[BaseException] = []
        for tracker_name, finish in (
            ("wandb", self._finish_wandb),
            ("mlflow", self._finish_mlflow),
        ):
            try:
                finish(exit_code=exit_code)
            except BaseException as exc:
                logger.exception("Failed to finalize %s tracking.", tracker_name)
                errors.append(exc)
        if errors:
            raise errors[0]

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
            "drop_last": self.dataloader_drop_last,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.dataloader_persistent_workers
            if self.dataloader_prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(
        self,
        scheduler_type,
        total_train_steps: int,
        warmup_steps: int = 0,
        optimizer=None,
    ):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
        optimizer = self.optimizer if optimizer is None else optimizer

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _reset_scheduler_after_state_resume(self):
        if not self.reset_scheduler_on_state_resume:
            return

        remaining_steps = max(int(self.max_steps) - int(self.global_step), 0)
        scheduler_steps = (
            remaining_steps
            if self.resume_scheduler_steps is None
            else int(self.resume_scheduler_steps)
        )
        if scheduler_steps < 1:
            raise ValueError(
                "Cannot reset the scheduler after state resume because the configured "
                f"continuation has no optimizer steps: max_steps={self.max_steps}, "
                f"global_step={self.global_step}, resume_scheduler_steps={self.resume_scheduler_steps}."
            )
        if scheduler_steps != remaining_steps:
            raise ValueError(
                "resume_scheduler_steps must equal max_steps - restored global_step so the "
                "continuation schedule terminates with training: "
                f"resume_scheduler_steps={scheduler_steps}, remaining_steps={remaining_steps}."
            )

        warmup_steps = int(scheduler_steps * self.resume_scheduler_warmup_fraction)
        accelerated_scheduler = self.scheduler
        raw_scheduler = getattr(accelerated_scheduler, "scheduler", accelerated_scheduler)
        raw_optimizer = raw_scheduler.optimizer

        optimizer_candidates = [raw_optimizer, self.optimizer]
        optimizer_candidates.extend(getattr(accelerated_scheduler, "optimizers", []))
        seen_optimizers = set()
        for optimizer in optimizer_candidates:
            if optimizer is None or id(optimizer) in seen_optimizers:
                continue
            seen_optimizers.add(id(optimizer))
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.learning_rate
                param_group["initial_lr"] = self.learning_rate

        replacement = self._build_scheduler(
            scheduler_type=self.cfg.lr_scheduler_type,
            total_train_steps=scheduler_steps,
            warmup_steps=warmup_steps,
            optimizer=raw_optimizer,
        )
        if hasattr(accelerated_scheduler, "scheduler"):
            accelerated_scheduler.scheduler = replacement
        else:
            self.scheduler = replacement

        resumed_lr = float(raw_optimizer.param_groups[0]["lr"])
        for optimizer in optimizer_candidates:
            if optimizer is None:
                continue
            for param_group in optimizer.param_groups:
                param_group["lr"] = resumed_lr
                param_group["initial_lr"] = self.learning_rate
        logger.warning(
            "Reset scheduler after full-state resume: restored_step=%d continuation_steps=%d "
            "scheduler_type=%s base_lr=%.8g initial_lr=%.8g warmup_steps=%d. "
            "Model/optimizer/RNG/dataloader state remain restored; the exhausted parent "
            "scheduler state is intentionally replaced.",
            self.global_step,
            scheduler_steps,
            self.cfg.lr_scheduler_type,
            self.learning_rate,
            resumed_lr,
            warmup_steps,
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        if self._weight_checkpoint_loaded_before_prepare:
            logger.info(
                "Weight checkpoint was loaded before optimizer/DeepSpeed initialization; "
                "skipping post-prepare reload."
            )
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _load_weight_checkpoint_before_prepare(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info(
            "Loading weight checkpoint before optimizer/DeepSpeed initialization: %s",
            resume,
        )
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weight_checkpoint_loaded_before_prepare = True

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        extra_trainable_modules = getattr(model, "extra_trainable_modules", None)
        if callable(extra_trainable_modules):
            for module in extra_trainable_modules():
                module.train()
                module.requires_grad_(True)
        apply_trainable_overrides = getattr(model, "apply_trainable_overrides", None)
        if callable(apply_trainable_overrides):
            apply_trainable_overrides()

    @staticmethod
    def _collect_trainable_parameters(model):
        trainable_params = []
        seen = set()

        def add_parameters(parameters):
            for parameter in parameters:
                if not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                trainable_params.append(parameter)

        add_parameters(model.dit.parameters())
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            add_parameters(proprio_encoder.parameters())
        extra_trainable_parameters = getattr(model, "extra_trainable_parameters", None)
        if callable(extra_trainable_parameters):
            add_parameters(extra_trainable_parameters())
        return trainable_params

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    @staticmethod
    def _normalize_save_at_steps(save_at_steps) -> frozenset[int]:
        if save_at_steps is None:
            return frozenset()
        if isinstance(save_at_steps, (str, bytes)) or not hasattr(save_at_steps, "__iter__"):
            raise TypeError(
                "save_at_steps must be a sequence of positive optimizer-step integers, "
                f"got {save_at_steps!r}."
            )
        normalized = frozenset(int(step) for step in save_at_steps)
        invalid = sorted(step for step in normalized if step <= 0)
        if invalid:
            raise ValueError(
                "save_at_steps must contain only positive optimizer-step integers, "
                f"got invalid values: {invalid}."
            )
        return normalized

    def _checkpoint_reason(self) -> str | None:
        reasons = []
        if self.save_every > 0 and self.global_step % self.save_every == 0:
            reasons.append("periodic")
        if self.global_step in self.save_at_steps:
            reasons.append("explicit")
        return "+".join(reasons) if reasons else None

    def _completion_checkpoint(self, current_step_checkpoint):
        """Reuse a checkpoint already written at the terminating optimizer step."""
        if current_step_checkpoint is not None:
            return current_step_checkpoint
        if self.save_final:
            return self.save_checkpoint()
        return {"weights_path": None, "state_path": None}

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _prune_training_states(self):
        keep = self.max_training_state_checkpoints
        if keep is None:
            return

        state_root = Path(self.state_dir)
        checkpoints = []
        for path in state_root.iterdir():
            match = re.fullmatch(r"step_(\d+)", path.name)
            if path.is_dir() and match is not None:
                checkpoints.append((int(match.group(1)), path))
        checkpoints.sort(key=lambda item: item[0])

        for _, path in checkpoints[:-keep]:
            logger.info(
                "Pruning training state checkpoint due to retention=%d: %s",
                keep,
                path,
            )
            shutil.rmtree(path)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = None
        if self.save_training_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._prune_training_states()
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self._reset_scheduler_after_state_resume()
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self._reset_scheduler_after_state_resume()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            profile_step = (
                self.trainer_profile_steps > 0 and self.global_step < self.trainer_profile_steps
            )
            rank_profile = self.rank_profile_steps > 0 and self.global_step < self.rank_profile_steps
            iter_start = time.perf_counter()
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            data_time = time.perf_counter() - iter_start

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
                if hasattr(train_model, "set_train_step"):
                    train_model.set_train_step(self.global_step)

                forward_start = time.perf_counter()
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                if not torch.isfinite(loss).all():
                    debug_losses = {}
                    for key, value in loss_dict.items():
                        if torch.is_tensor(value):
                            debug_losses[key] = float(value.detach().float().mean().cpu())
                        else:
                            debug_losses[key] = float(value)
                    raise RuntimeError(
                        f"Non-finite training loss at global_step={self.global_step}: "
                        f"loss={float(loss.detach().float().mean().cpu())}, loss_dict={debug_losses}"
                    )
                if (profile_step or rank_profile) and torch.cuda.is_available():
                    torch.cuda.synchronize(self.accelerator.device)
                forward_time = time.perf_counter() - forward_start

                backward_start = time.perf_counter()
                self.accelerator.backward(loss)
                if (profile_step or rank_profile) and torch.cuda.is_available():
                    torch.cuda.synchronize(self.accelerator.device)
                backward_time = time.perf_counter() - backward_start
                sync_gradients = bool(self.accelerator.sync_gradients)

                if sync_gradients:
                    opt_start = time.perf_counter()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if (profile_step or rank_profile) and torch.cuda.is_available():
                        torch.cuda.synchronize(self.accelerator.device)
                    opt_time = time.perf_counter() - opt_start
                    if rank_profile:
                        self._write_rank_profile(
                            {
                                "event": "microstep",
                                "global_step": int(self.global_step),
                                "batch_in_epoch": int(self.batch_in_epoch),
                                "sync_gradients": sync_gradients,
                                "data": round(data_time, 6),
                                "forward": round(forward_time, 6),
                                "backward": round(backward_time, 6),
                                "backward_includes_deepspeed_step": (
                                    self._backward_includes_optimizer_step(sync_gradients)
                                ),
                                "optimizer": round(opt_time, 6),
                                "total": round(time.perf_counter() - iter_start, 6),
                                "loss": float(loss.detach().float().item()),
                            }
                        )
                    if profile_step:
                        logger.info(
                            "[profile:trainer] step=%d data=%.4f forward=%.4f backward=%.4f "
                            "backward_includes_deepspeed_step=%s optimizer_wrapper=%.4f total=%.4f",
                            self.global_step,
                            data_time,
                            forward_time,
                            backward_time,
                            self._backward_includes_optimizer_step(sync_gradients),
                            opt_time,
                            time.perf_counter() - iter_start,
                        )
                    self.global_step += 1
                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    should_log = self.log_every > 0 and self.global_step % self.log_every == 0

                    if should_log:
                        gathered_metrics = self._gather_scalar_metrics(
                            {
                                "__loss__": loss,
                                "__grad_norm__": grad_norm,
                                **loss_dict,
                            },
                            device=loss.device,
                        )
                        global_loss = gathered_metrics.pop("__loss__")
                        global_grad_norm = gathered_metrics.pop("__grad_norm__")
                        global_loss_metrics = gathered_metrics

                    if should_log and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join(
                                [f"{k}={v:.8g}" for k, v in sorted(global_loss_metrics.items())]
                            )
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            self._samples_per_second(steps_per_sec),
                            eta_str,
                        )
                        logger.info(description)

                        tracking_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": self._samples_per_second(steps_per_sec),
                            "performance/effective_global_batch_size": self._effective_global_batch_size(),
                        }
                        tracking_payload.update(
                            self._tracking_runtime_metrics(
                                steps_per_sec=steps_per_sec,
                                data_time=data_time,
                                forward_time=forward_time,
                                backward_time=backward_time,
                                optimizer_time=opt_time,
                                iteration_time=time.perf_counter() - iter_start,
                                optimizer_step_skipped=bool(
                                    self.accelerator.optimizer_step_was_skipped
                                ),
                            )
                        )
                        for key, value in global_loss_metrics.items():
                            tracking_payload[f"train/{key}"] = value
                        self._log_tracking_metrics(tracking_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._log_tracking_metrics(eval_payload)

                    checkpoint_reason = self._checkpoint_reason()
                    current_step_checkpoint = None
                    if checkpoint_reason is not None:
                        current_step_checkpoint = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d reason=%s weights=%s state=%s",
                                self.global_step,
                                checkpoint_reason,
                                current_step_checkpoint["weights_path"],
                                current_step_checkpoint["state_path"],
                            )
                            self._record_checkpoint(current_step_checkpoint)

                    if self.global_step >= self.max_steps:
                        ckpt_info = self._completion_checkpoint(current_step_checkpoint)
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                            self._record_checkpoint(ckpt_info)
                        return
                else:
                    if rank_profile:
                        self._write_rank_profile(
                            {
                                "event": "microstep",
                                "global_step": int(self.global_step),
                                "batch_in_epoch": int(self.batch_in_epoch),
                                "sync_gradients": sync_gradients,
                                "data": round(data_time, 6),
                                "forward": round(forward_time, 6),
                                "backward": round(backward_time, 6),
                                "backward_includes_deepspeed_step": (
                                    self._backward_includes_optimizer_step(sync_gradients)
                                ),
                                "optimizer": None,
                                "total": round(time.perf_counter() - iter_start, 6),
                                "loss": float(loss.detach().float().item()),
                            }
                        )

        ckpt_info = self._completion_checkpoint(None)
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
            self._record_checkpoint(ckpt_info)
        
