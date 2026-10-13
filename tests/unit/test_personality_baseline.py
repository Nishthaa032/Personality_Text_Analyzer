from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.personality.tfidf_regression_baseline import TRAITS, train_and_evaluate

RESULTS_PATH = ROOT / "research" / "personality_tfidf_baseline.json"


def test_baseline_training_produces_metrics_and_model_report():
    report = train_and_evaluate()

    assert "selected_alpha" in report
    assert "metrics" in report
    assert len(report["metrics"]) == 2
    assert set(report["metrics"][0].keys()) >= {"split", "mae_by_trait", "rmse_by_trait", "pearson_by_trait"}

    assert RESULTS_PATH.exists()
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assert data["trait_columns"] == TRAITS
    assert len(data["metrics"]) == 2


def test_baseline_predictions_are_numeric_and_in_range():
    df = pd.read_csv(ROOT / "data" / "processed" / "eval_processed.csv")

    metrics = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    eval_metrics = next(item for item in metrics["metrics"] if item["split"] == "eval")

    for trait in TRAITS:
        assert trait in eval_metrics["mae_by_trait"]
        assert trait in eval_metrics["rmse_by_trait"]
        assert trait in eval_metrics["pearson_by_trait"]
        assert np.isfinite(eval_metrics["mae_by_trait"][trait])
        assert np.isfinite(eval_metrics["rmse_by_trait"][trait])
        assert np.isfinite(eval_metrics["pearson_by_trait"][trait])

    assert len(df) > 0
