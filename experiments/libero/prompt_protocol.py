"""Prompt selection for LIBERO-Plus tasks using the training instruction map."""
import json
import hashlib
import re
from functools import lru_cache
from pathlib import Path

NONLANGUAGE_CATEGORIES = {
    "Background Textures", "Camera Viewpoints", "Robot Initial States",
    "Sensor Noise", "Objects Layout", "Light Conditions",
}


@lru_cache(maxsize=8)
def _load_mapping(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported training instruction map schema")
    return data["tasks"]


def get_training_prompt_entry(task, mapping_path):
    if not mapping_path:
        raise ValueError("training_nonlanguage requires EVALUATION.training_instruction_map")
    try:
        entry = _load_mapping(str(mapping_path))[str(task.problem_folder)][str(task.name)]
    except KeyError as exc:
        raise ValueError(f"Task absent from validated instruction map: {task.problem_folder}/{task.name}") from exc
    category = entry["category"]
    if category == "Language Instructions":
        if "_language_" not in str(task.name):
            raise ValueError("Language category and task filename disagree")
        return entry  # the caller reads the instruction from the variant BDDL
    if category not in NONLANGUAGE_CATEGORIES or "_language_" in str(task.name):
        raise ValueError(f"Invalid non-language prompt mapping: {task.name}, {category}")
    if not isinstance(entry.get("instruction"), str) or not entry["instruction"].strip():
        raise ValueError(f"Empty training instruction: {task.name}")
    return entry


def build_training_instruction_map(dataset_root, classification_path, bddl_root):
    """Map each LIBERO-Plus task name to its training instruction."""
    metadata = json.loads(Path(classification_path).read_text())
    mapping, hashes = {}, {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        corpus = Path(dataset_root) / f"{suite}_no_noops_lerobot/meta/tasks.jsonl"
        hashes[str(corpus)] = hashlib.sha256(corpus.read_bytes()).hexdigest()
        instructions = [json.loads(line)["task"] for line in corpus.read_text().splitlines()]
        bases = {}
        files = list((Path(bddl_root) / suite).glob("*.bddl"))
        for instruction in instructions:
            matches = [p.stem for p in files if re.sub(r"^.*_SCENE\d+_", "", p.stem).lower() == instruction.replace(" ", "_")]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous/missing training base: {suite}/{instruction}: {matches}")
            bases[matches[0]] = instruction
        mapping[suite] = {}
        for task in metadata[suite]:
            name, category = task["name"], task["category"]
            if ("_language_" in name) != (category == "Language Instructions"):
                raise ValueError(f"Language metadata disagreement: {name}")
            if category == "Language Instructions":
                mapping[suite][name] = {"category": category, "instruction": None}
                continue
            if category not in NONLANGUAGE_CATEGORIES:
                raise ValueError(f"Unknown category: {category}")
            matches = [base for base in bases if name == base or name.startswith(base + "_")]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous/missing variant base: {suite}/{name}: {matches}")
            base = matches[0]
            mapping[suite][name] = {"category": category, "base_task": base, "instruction": bases[base]}
    return {"schema_version": 1, "tasks": mapping, "training_source_sha256": hashes,
            "classification_sha256": hashlib.sha256(Path(classification_path).read_bytes()).hexdigest()}
