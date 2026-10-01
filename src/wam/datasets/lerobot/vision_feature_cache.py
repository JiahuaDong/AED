from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


CACHE_FORMAT_VERSION = 1


def tensor_to_bfloat16_bits(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return tensor.view(torch.uint16).numpy()


def bfloat16_bits_to_tensor(array: np.ndarray) -> torch.Tensor:
    copied = np.array(array, dtype=np.uint16, copy=True, order="C")
    return torch.from_numpy(copied).view(torch.bfloat16)


class VisionFeatureCache:
    """Read per-sample VAE/DINO arrays from memory-mapped NPY files."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_length: int,
        expected_spec: dict[str, Any],
        expected_feature_spec: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing vision feature cache manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if int(self.manifest.get("format_version", -1)) != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported vision cache format in {manifest_path}: "
                f"{self.manifest.get('format_version')!r}."
            )
        if not bool(self.manifest.get("complete", False)):
            raise RuntimeError(f"Vision feature cache is incomplete: {manifest_path}")
        if int(self.manifest.get("dataset_length", -1)) != int(dataset_length):
            raise ValueError(
                "Vision feature cache dataset length mismatch: "
                f"cache={self.manifest.get('dataset_length')} current={dataset_length}."
            )
        cached_spec = self.manifest.get("dataset_spec", {})
        if cached_spec != expected_spec:
            raise ValueError(
                "Vision feature cache dataset specification mismatch. "
                f"cache={cached_spec}, current={expected_spec}."
            )
        if expected_feature_spec is not None:
            cached_feature_spec = self.manifest.get("feature_spec")
            if not isinstance(cached_feature_spec, dict):
                raise ValueError(
                    "Vision feature cache has no frozen-target signature, but the current "
                    f"dataset requires one: {manifest_path}."
                )
            mismatches = {
                key: {
                    "cache": cached_feature_spec.get(key),
                    "expected": expected_value,
                }
                for key, expected_value in expected_feature_spec.items()
                if cached_feature_spec.get(key) != expected_value
            }
            if mismatches:
                raise ValueError(
                    "Vision feature cache frozen-target signature mismatch: "
                    f"{mismatches}. Cache={manifest_path}."
                )
        arrays = self.manifest.get("arrays", {})
        if not arrays:
            raise ValueError(f"Vision feature cache contains no arrays: {manifest_path}")
        self.array_specs = dict(arrays)
        self._arrays: dict[tuple[str, tuple[int, ...], str], np.ndarray] = {}
        self._indices: dict[str, np.ndarray] = {}

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.array_specs)

    def _array(self, key: str) -> np.ndarray:
        spec = self.array_specs[key]
        path = self.root / str(spec["file"])
        expected_shape = tuple(int(x) for x in spec["shape"])
        storage_format = str(spec.get("format", "npy"))
        storage_key = (str(path), expected_shape, storage_format)
        array = self._arrays.get(storage_key)
        if array is not None:
            return array
        if spec.get("storage_dtype") != "bfloat16_bits":
            raise ValueError(f"Unsupported storage dtype for cache array {key!r}: {spec.get('storage_dtype')!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing vision cache array {key!r}: {path}")
        if storage_format == "npy":
            array = np.load(path, mmap_mode="r")
        elif storage_format == "raw":
            array = np.memmap(path, mode="r", dtype=np.uint16, shape=expected_shape)
        else:
            raise ValueError(f"Unsupported vision cache storage format for {key!r}: {storage_format!r}")
        if tuple(array.shape) != expected_shape or array.dtype != np.uint16:
            raise ValueError(
                f"Vision cache array {key!r} mismatch: got shape={array.shape} dtype={array.dtype}, "
                f"expected shape={expected_shape} dtype=uint16."
            )
        self._arrays[storage_key] = array
        return array

    def _index_array(self, key: str) -> np.ndarray | None:
        spec = self.array_specs[key]
        index_file = spec.get("index_file")
        if index_file is None:
            return None
        index_path = self.root / str(index_file)
        cache_key = str(index_path)
        indices = self._indices.get(cache_key)
        if indices is not None:
            return indices
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing vision cache index for {key!r}: {index_path}")
        indices = np.load(index_path, mmap_mode="r")
        expected_shape = tuple(int(x) for x in spec["index_shape"])
        if tuple(indices.shape) != expected_shape or indices.dtype != np.int32:
            raise ValueError(
                f"Vision cache index {key!r} mismatch: got shape={indices.shape} dtype={indices.dtype}, "
                f"expected shape={expected_shape} dtype=int32."
            )
        self._indices[cache_key] = indices
        return indices

    def get(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= int(self.manifest["dataset_length"]):
            raise IndexError(f"Vision cache index {index} is out of range.")
        result = {}
        for key in self.array_specs:
            array = self._array(key)
            indices = self._index_array(key)
            value = array[index] if indices is None else array[indices[index]]
            result[key] = bfloat16_bits_to_tensor(value)
        return result
