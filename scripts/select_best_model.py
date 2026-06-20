"""
Rank trained experiments and select the best model.

Expected CSV columns:
  experiment,speaker_similarity,cer,wer,mos,latency_sec,notes

Example:
  python scripts/select_best_model.py
  python scripts/select_best_model.py --metrics-csv evaluation/model_scores.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COLUMNS = [
    "experiment",
    "speaker_similarity",
    "cer",
    "wer",
    "mos",
    "latency_sec",
    "notes",
]


def load_experiment_names(matrix_path: Path) -> List[str]:
    with matrix_path.open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f)
    return [exp["name"] for exp in matrix.get("experiments", [])]


def create_template(metrics_csv: Path, experiment_names: List[str]) -> None:
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        for name in experiment_names:
            writer.writerow({
                "experiment": name,
                "speaker_similarity": "",
                "cer": "",
                "wer": "",
                "mos": "",
                "latency_sec": "",
                "notes": "",
            })


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    value = (row.get(key) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_row(row: Dict[str, str]) -> Optional[float]:
    similarity = parse_float(row, "speaker_similarity")
    cer = parse_float(row, "cer")
    wer = parse_float(row, "wer")
    mos = parse_float(row, "mos")
    latency = parse_float(row, "latency_sec")

    required = [similarity, cer, wer, mos, latency]
    if any(value is None for value in required):
        return None

    similarity_score = clamp(similarity)
    cer_score = 1.0 - clamp(cer)
    wer_score = 1.0 - clamp(wer)
    mos_score = clamp(mos / 5.0)
    latency_score = 1.0 / (1.0 + max(latency, 0.0) / 10.0)

    return (
        0.35 * similarity_score
        + 0.25 * cer_score
        + 0.15 * wer_score
        + 0.15 * mos_score
        + 0.10 * latency_score
    )


def load_scores(metrics_csv: Path) -> List[Dict[str, str]]:
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_report(output_path: Path, ranked_rows: List[Dict[str, str]], best: Dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Selection Report",
        "",
        f"Best experiment: **{best['experiment']}**",
        f"Overall score: **{float(best['overall_score']):.4f}**",
        "",
        "| Rank | Experiment | Score | Similarity | CER | WER | MOS | Latency |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked_rows, start=1):
        lines.append(
            "| {rank} | {experiment} | {score:.4f} | {similarity} | {cer} | {wer} | {mos} | {latency} |".format(
                rank=idx,
                experiment=row["experiment"],
                score=float(row["overall_score"]),
                similarity=row.get("speaker_similarity", ""),
                cer=row.get("cer", ""),
                wer=row.get("wer", ""),
                mos=row.get("mos", ""),
                latency=row.get("latency_sec", ""),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_json(output_path: Path, best: Dict[str, str]) -> None:
    checkpoint_dir = f"checkpoints/experiments/{best['experiment']}"
    payload = {
        "experiment": best["experiment"],
        "overall_score": float(best["overall_score"]),
        "checkpoint_dir": checkpoint_dir,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the best F5-TTS experiment")
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--metrics-csv", default="evaluation/model_scores.csv")
    parser.add_argument("--report", default="evaluation/best_model_report.md")
    parser.add_argument("--best-json", default="evaluation/best_model.json")
    args = parser.parse_args()

    matrix_path = PROJECT_DIR / args.matrix
    metrics_csv = PROJECT_DIR / args.metrics_csv
    report_path = PROJECT_DIR / args.report
    best_json_path = PROJECT_DIR / args.best_json

    if not metrics_csv.exists():
        create_template(metrics_csv, load_experiment_names(matrix_path))
        print(f"Đã tạo template metric: {metrics_csv}")
        print("Hãy điền metric sau khi test audio rồi chạy lại script này.")
        return 0

    rows = load_scores(metrics_csv)
    scored = []
    skipped = []
    for row in rows:
        score = score_row(row)
        if score is None:
            skipped.append(row.get("experiment", "<unknown>"))
            continue
        row = dict(row)
        row["overall_score"] = f"{score:.8f}"
        scored.append(row)

    if not scored:
        raise SystemExit(
            "Chưa có dòng metric đầy đủ để chọn model. "
            "Cần speaker_similarity, cer, wer, mos, latency_sec."
        )

    ranked = sorted(scored, key=lambda item: float(item["overall_score"]), reverse=True)
    best = ranked[0]
    write_report(report_path, ranked, best)
    write_best_json(best_json_path, best)

    print(f"Best experiment: {best['experiment']} ({float(best['overall_score']):.4f})")
    print(f"Report         : {report_path}")
    print(f"Best model JSON: {best_json_path}")
    if skipped:
        print("Skipped incomplete rows:", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
