"""
Project preflight checks that do not require GPU or F5-TTS runtime imports.

Use this before uploading/running on Colab:
  python scripts/preflight_check.py
"""

import ast
import csv
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "app.py",
    "verify_env.py",
    "setup.sh",
    "requirements.txt",
    "configs/train_config.yaml",
    "configs/experiment_matrix.yaml",
    "scripts/data_prep.py",
    "scripts/train.py",
    "scripts/run_experiments.py",
    "scripts/select_best_model.py",
]
METRIC_COLUMNS = [
    "experiment",
    "speaker_similarity",
    "cer",
    "wer",
    "mos",
    "latency_sec",
    "notes",
]


def load_yaml(relative_path: str) -> Dict[str, Any]:
    with (PROJECT_DIR / relative_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_required_files(errors: List[str]) -> None:
    for relative_path in REQUIRED_FILES:
        if not (PROJECT_DIR / relative_path).exists():
            errors.append(f"Missing required file: {relative_path}")


def check_python_ast(errors: List[str]) -> None:
    for path in PROJECT_DIR.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"Python syntax error in {path.relative_to(PROJECT_DIR)}: {exc}")


def check_t4_training_config(config: Dict[str, Any], label: str, errors: List[str]) -> None:
    train_cfg = config.get("training", {})
    audio_cfg = config.get("audio", {})

    if train_cfg.get("mixed_precision") != "fp16":
        errors.append(f"{label}: mixed_precision must be fp16 for NVIDIA T4")
    if int(train_cfg.get("batch_size", 999)) > 2:
        errors.append(f"{label}: batch_size must be <= 2 for NVIDIA T4")
    if int(train_cfg.get("gradient_accumulation_steps", 0)) < 1:
        errors.append(f"{label}: gradient_accumulation_steps must be >= 1")
    if float(train_cfg.get("learning_rate", 0.0)) <= 0:
        errors.append(f"{label}: learning_rate must be > 0")
    if audio_cfg.get("sample_rate") != 24000:
        errors.append(f"{label}: sample_rate must be 24000")
    if audio_cfg.get("hop_length") != 256:
        errors.append(f"{label}: hop_length must be 256")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def check_experiment_matrix(errors: List[str]) -> None:
    base = load_yaml("configs/train_config.yaml")
    matrix = load_yaml("configs/experiment_matrix.yaml")
    experiments = matrix.get("experiments", [])

    if not (3 <= len(experiments) <= 6):
        errors.append(f"Experiment count must be 3-6, got {len(experiments)}")

    names = [exp.get("name") for exp in experiments]
    if len(names) != len(set(names)):
        errors.append("Experiment names must be unique")

    for exp in experiments:
        name = exp.get("name", "<missing>")
        if not name or name == "<missing>":
            errors.append("Experiment missing name")
            continue
        config = deep_merge(base, exp.get("overrides", {}))
        check_t4_training_config(config, name, errors)

        ckpt_dir = f"checkpoints/experiments/{name}"
        log_dir = f"logs/experiments/{name}"
        if ckpt_dir == "checkpoints/experiments/":
            errors.append(f"{name}: invalid checkpoint dir")
        if log_dir == "logs/experiments/":
            errors.append(f"{name}: invalid log dir")


def check_metrics_template(errors: List[str]) -> None:
    metrics_path = PROJECT_DIR / "evaluation" / "model_scores.csv"
    if not metrics_path.exists():
        return

    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != METRIC_COLUMNS:
            errors.append(f"evaluation/model_scores.csv columns must be: {METRIC_COLUMNS}")


def main() -> int:
    errors: List[str] = []
    check_required_files(errors)
    check_python_ast(errors)
    check_t4_training_config(load_yaml("configs/train_config.yaml"), "base_config", errors)
    check_experiment_matrix(errors)
    check_metrics_template(errors)

    if errors:
        print("PRE-FLIGHT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PRE-FLIGHT PASSED")
    print("- Required files exist")
    print("- Python files parse correctly")
    print("- YAML configs are valid")
    print("- 3-6 experiment requirement is satisfied")
    print("- T4 safety constraints are satisfied")
    print("- Metric template shape is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
