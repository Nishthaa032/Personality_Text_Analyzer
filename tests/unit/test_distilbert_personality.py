from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.personality.distilbert_regression import TRAITS, train_and_evaluate

RESULTS_PATH = ROOT / "research" / "personality_distilbert_regression.json"


def test_distilbert_training_creates_report_and_metrics():
    report = train_and_evaluate(epochs=1, batch_size=2, max_train_samples=50, max_val_samples=20, max_eval_samples=20)

    assert "metrics" in report
    assert len(report["metrics"]) == 2
    assert set(report["metrics"][0].keys()) >= {"split", "mae_by_trait", "rmse_by_trait", "pearson_by_trait"}
    assert RESULTS_PATH.exists()

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assert data["trait_columns"] == TRAITS
    assert len(data["metrics"]) == 2


def test_distilbert_report_contains_all_traits():
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    for split in data["metrics"]:
        for trait in TRAITS:
            assert trait in split["mae_by_trait"]
            assert trait in split["rmse_by_trait"]
            assert trait in split["pearson_by_trait"]
