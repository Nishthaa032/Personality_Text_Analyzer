from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
TRAITS = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}_processed.csv")


def test_processed_splits_have_expected_columns():
    for split_name in ["train", "val", "eval"]:
        df = _load_split(split_name)
        assert "text" in df.columns
        for trait in TRAITS:
            assert trait in df.columns


def test_processed_splits_are_overlap_free():
    train_df = _load_split("train")
    val_df = _load_split("val")
    eval_df = _load_split("eval")

    assert not set(train_df["text"]) & set(val_df["text"])
    assert not set(train_df["text"]) & set(eval_df["text"])
    assert not set(val_df["text"]) & set(eval_df["text"])


def test_processed_splits_have_no_duplicate_texts():
    for split_name in ["train", "val", "eval"]:
        df = _load_split(split_name)
        assert df["text"].duplicated().sum() == 0


def test_processed_splits_stay_in_valid_score_range():
    for split_name in ["train", "val", "eval"]:
        df = _load_split(split_name)
        for trait in TRAITS:
            values = pd.to_numeric(df[trait], errors="raise")
            assert (values >= 0).all()
            assert (values <= 100).all()
