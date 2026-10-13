from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np

TRAITS = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "personality_tfidf_ridge.joblib"
VECTORIZER_PATH = MODEL_DIR / "personality_tfidf_vectorizer.joblib"


class PersonalityPredictor:
    def __init__(self, model_path: str | Path = MODEL_PATH, vectorizer_path: str | Path = VECTORIZER_PATH):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict_text(self, text: str) -> dict[str, float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        prediction = self.model.predict(self.vectorizer.transform([text]))[0]
        return {trait: float(value) for trait, value in zip(TRAITS, prediction)}

    def predict_batch(self, texts: Sequence[str]) -> list[dict[str, float]]:
        cleaned = [str(text).strip() for text in texts if str(text).strip()]
        if not cleaned:
            return []
        predictions = self.model.predict(self.vectorizer.transform(cleaned))
        return [
            {trait: float(value) for trait, value in zip(TRAITS, row)}
            for row in predictions
        ]


def load_predictor() -> PersonalityPredictor:
    return PersonalityPredictor()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Predict Big Five personality scores from raw text.")
    parser.add_argument("text", nargs="+", help="Text to score.")
    args = parser.parse_args()

    predictor = load_predictor()
    text = " ".join(args.text)
    result = predictor.predict_text(text)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()
