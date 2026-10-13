from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.personality.inference import TRAITS, PersonalityPredictor


def test_predictor_returns_all_trait_scores_for_single_text():
    predictor = PersonalityPredictor()
    result = predictor.predict_text("I feel energized and excited to talk with friends and make plans for the weekend.")

    assert set(result.keys()) == set(TRAITS)
    for trait, score in result.items():
        assert isinstance(trait, str)
        assert 0.0 <= score <= 100.0, f"{trait} score out of expected range: {score}"


def test_predictor_returns_batch_predictions():
    predictor = PersonalityPredictor()
    texts = [
        "I like to stay organized and plan carefully before I act.",
        "I enjoy spending time with friends and being socially active.",
    ]

    results = predictor.predict_batch(texts)
    assert len(results) == 2
    for result in results:
        assert set(result.keys()) == set(TRAITS)
        assert all(0.0 <= value <= 100.0 for value in result.values())
