from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "research"
BASELINE_PATH = RESEARCH_DIR / "personality_tfidf_baseline.json"
DISTILBERT_PATH = RESEARCH_DIR / "personality_distilbert_regression.json"
OUTPUT_PATH = RESEARCH_DIR / "model_comparison_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_split(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_mae": float(metrics["mean_mae"]),
        "mean_rmse": float(metrics["mean_rmse"]),
        "mean_pearson": float(metrics["mean_pearson"]),
    }


def build_summary() -> dict[str, Any]:
    baseline_report = load_json(BASELINE_PATH)
    distilbert_report = load_json(DISTILBERT_PATH)

    baseline_val = next(item for item in baseline_report["metrics"] if item["split"] == "val")
    baseline_eval = next(item for item in baseline_report["metrics"] if item["split"] == "eval")
    distilbert_val = next(item for item in distilbert_report["metrics"] if item["split"] == "val")
    distilbert_eval = next(item for item in distilbert_report["metrics"] if item["split"] == "eval")

    def choose_better(baseline: dict[str, float], transformer: dict[str, float]) -> str:
        if baseline["mean_mae"] < transformer["mean_mae"] and baseline["mean_pearson"] > transformer["mean_pearson"]:
            return "TF-IDF + Ridge"
        return "DistilBERT"

    summary = {
        "project_scope": "personality-only",
        "verified_constraints": [
            "The workspace contains only a personality dataset; emotion labels are absent.",
            "The emotion task remains deferred until the exact GoEmotions dataset and schema are provided.",
        ],
        "models": {
            "tfidf_ridge": {
                "description": "TF-IDF + Ridge regression baseline",
                "validation": summarize_split(baseline_val),
                "evaluation": summarize_split(baseline_eval),
            },
            "distilbert": {
                "description": "DistilBERT regression candidate",
                "validation": summarize_split(distilbert_val),
                "evaluation": summarize_split(distilbert_eval),
            },
        },
        "winner_by_validation": {
            "model": choose_better(summarize_split(baseline_val), summarize_split(distilbert_val)),
            "reason": "Lower MAE and higher Pearson correlation are preferred for this regression task.",
        },
        "winner_by_evaluation": {
            "model": choose_better(summarize_split(baseline_eval), summarize_split(distilbert_eval)),
            "reason": "Lower MAE and higher Pearson correlation are preferred for this regression task.",
        },
        "conclusion": (
            "On the verified personality dataset in this workspace, the TF-IDF + Ridge baseline remains the strongest model "
            "under current evidence; the DistilBERT candidate is functional but does not outperform the baseline on MAE or Pearson."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    report = build_summary()
    print(json.dumps(report, indent=2))
