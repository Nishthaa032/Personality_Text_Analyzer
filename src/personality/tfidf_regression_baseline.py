from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRAITS = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "research"


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if np.isclose(denom, 0.0):
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    return pd.read_csv(path)


def select_alpha(train_df: pd.DataFrame, val_df: pd.DataFrame, candidates: List[float]) -> float:
    best_alpha = candidates[0]
    best_score = float("inf")

    for alpha in candidates:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(train_df["text"])
        X_val = vectorizer.transform(val_df["text"])
        y_train = train_df[TRAITS].to_numpy(dtype=float)
        y_val = val_df[TRAITS].to_numpy(dtype=float)

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds, multioutput="raw_values")
        score = float(np.mean(mae))

        if score < best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha


def train_and_evaluate() -> Dict[str, object]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    eval_df = load_split("eval")

    candidate_alphas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    alpha = select_alpha(train_df, val_df, candidate_alphas)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_eval = vectorizer.transform(eval_df["text"])

    y_train = train_df[TRAITS].to_numpy(dtype=float)
    y_val = val_df[TRAITS].to_numpy(dtype=float)
    y_eval = eval_df[TRAITS].to_numpy(dtype=float)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    eval_preds = model.predict(X_eval)

    metric_rows = []
    for name, y_true, y_pred in [("val", y_val, val_preds), ("eval", y_eval, eval_preds)]:
        mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
        pearson = np.array([pearson_r(y_true[:, idx], y_pred[:, idx]) for idx in range(len(TRAITS))])
        metric_rows.append(
            {
                "split": name,
                "alpha": float(alpha),
                "mae_by_trait": {trait: float(score) for trait, score in zip(TRAITS, mae)},
                "rmse_by_trait": {trait: float(score) for trait, score in zip(TRAITS, rmse)},
                "pearson_by_trait": {trait: float(score) for trait, score in zip(TRAITS, pearson)},
                "mean_mae": float(np.mean(mae)),
                "mean_rmse": float(np.mean(rmse)),
                "mean_pearson": float(np.mean(pearson)),
            }
        )

    model_path = MODEL_DIR / "personality_tfidf_ridge.joblib"
    vectorizer_path = MODEL_DIR / "personality_tfidf_vectorizer.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    report_path = RESULTS_DIR / "personality_tfidf_baseline.json"
    report = {
        "description": "TF-IDF + Ridge regression baseline for five Big Five personality traits",
        "selected_alpha": float(alpha),
        "candidates_tested": candidate_alphas,
        "trait_columns": TRAITS,
        "metrics": metric_rows,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    report = train_and_evaluate()
    print(json.dumps(report["metrics"], indent=2))
