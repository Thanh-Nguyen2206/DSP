"""
Run multiple fine-tuning experiments from configs/experiment_matrix.yaml.

Example:
  python scripts/run_experiments.py --dry-run
  python scripts/run_experiments.py --only m01_baseline_stable,m04_noise_robust
  python scripts/run_experiments.py
"""

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def parse_only(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def select_experiments(experiments: Iterable[Dict[str, Any]], only: List[str], limit: int) -> List[Dict[str, Any]]:
    selected = []
    only_set = set(only)
    for exp in experiments:
        if only_set and exp["name"] not in only_set:
            continue
        selected.append(exp)
        if limit and len(selected) >= limit:
            break
    return selected


def validate_generated_config(config: Dict[str, Any], exp_name: str) -> None:
    train_cfg = config["training"]
    if train_cfg.get("mixed_precision") == "bf16":
        raise ValueError(f"{exp_name}: T4 không hỗ trợ BF16, hãy dùng FP16.")
    if int(train_cfg.get("batch_size", 1)) > 2:
        raise ValueError(f"{exp_name}: batch_size > 2 dễ OOM trên T4.")
    if int(train_cfg.get("gradient_accumulation_steps", 1)) < 1:
        raise ValueError(f"{exp_name}: gradient_accumulation_steps phải >= 1.")


def build_experiment_config(base_config: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    exp_name = exp["name"]
    config = deep_merge(base_config, exp.get("overrides", {}))
    config["experiment"] = {
        "name": exp_name,
        "display_name": exp.get("display_name", exp_name),
        "description": exp.get("description", ""),
    }
    config.setdefault("checkpoint", {})
    config.setdefault("logging", {})
    config["checkpoint"]["save_dir"] = f"checkpoints/experiments/{exp_name}"
    config["logging"]["log_dir"] = f"logs/experiments/{exp_name}"
    validate_generated_config(config, exp_name)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run F5-TTS multi-model experiments")
    parser.add_argument(
        "--matrix",
        default="configs/experiment_matrix.yaml",
        help="Experiment matrix YAML path, relative to project dir by default.",
    )
    parser.add_argument(
        "--generated-dir",
        default="configs/generated",
        help="Directory for generated per-experiment configs.",
    )
    parser.add_argument("--only", default="", help="Comma-separated experiment names to run.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N selected experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue if one experiment fails.")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = PROJECT_DIR / matrix_path
    matrix = load_yaml(matrix_path)

    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        base_path = PROJECT_DIR / base_path
    base_config = load_yaml(base_path)

    selected = select_experiments(
        matrix.get("experiments", []),
        only=parse_only(args.only),
        limit=args.limit,
    )
    if not selected:
        raise SystemExit("Không có experiment nào được chọn.")

    generated_dir = Path(args.generated_dir)
    if not generated_dir.is_absolute():
        generated_dir = PROJECT_DIR / generated_dir

    print(f"Project dir : {PROJECT_DIR}")
    print(f"Matrix      : {matrix_path}")
    print(f"Experiments : {len(selected)}")

    for idx, exp in enumerate(selected, start=1):
        exp_name = exp["name"]
        generated_config = build_experiment_config(base_config, exp)
        generated_path = generated_dir / f"{exp_name}.yaml"
        write_yaml(generated_path, generated_config)

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(generated_path.relative_to(PROJECT_DIR)),
        ]
        print(f"\n[{idx}/{len(selected)}] {exp_name}")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=PROJECT_DIR)
        if result.returncode != 0:
            message = f"Experiment failed: {exp_name} (exit code {result.returncode})"
            if args.continue_on_error:
                print(message)
                continue
            raise SystemExit(message)

    print("\nHoàn tất danh sách experiment đã chọn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
