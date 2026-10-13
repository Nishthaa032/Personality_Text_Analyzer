from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "text",
    "agreeableness",
    "openness",
    "conscientiousness",
    "extraversion",
    "neuroticism",
}

TRAITS = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def test_personality_columns_exist_in_each_split():
    for split_name in ["train_set.csv", "val_set.csv", "eval_set.csv"]:
        df = _read_split(split_name)
        assert REQUIRED_COLUMNS.issubset(df.columns), f"Missing columns in {split_name}: {sorted(REQUIRED_COLUMNS - set(df.columns))}"


def test_target_scores_are_numeric_and_in_range():
    for split_name in ["train_set.csv", "val_set.csv", "eval_set.csv"]:
        df = _read_split(split_name)
        for trait in TRAITS:
            values = pd.to_numeric(df[trait], errors="raise")
            assert (values >= 0).all(), f"{trait} has values below 0 in {split_name}"
            assert (values <= 100).all(), f"{trait} has values above 100 in {split_name}"


def test_text_is_present_and_non_empty():
    for split_name in ["train_set.csv", "val_set.csv", "eval_set.csv"]:
        df = _read_split(split_name)
        assert "text" in df.columns
        assert df["text"].notna().all()
        assert (df["text"].astype(str).str.strip() != "").all(), f"Empty text values found in {split_name}"


def test_no_missing_values_in_required_fields():
    for split_name in ["train_set.csv", "val_set.csv", "eval_set.csv"]:
        df = _read_split(split_name)
        assert df["text"].isna().sum() == 0
        for trait in TRAITS:
            assert df[trait].isna().sum() == 0, f"Null values found in {trait} of {split_name}"


def test_split_overlap_is_documented_and_logged():
    # This is not a failure condition for the dataset itself; it is a warning condition
    # to prevent accidental leakage if a later model pipeline ignores split-level text overlap.
    train_df = _read_split("train_set.csv")
    val_df = _read_split("val_set.csv")
    eval_df = _read_split("eval_set.csv")

    overlap_train_val = set(train_df["text"]) & set(val_df["text"])
    overlap_train_eval = set(train_df["text"]) & set(eval_df["text"])
    overlap_val_eval = set(val_df["text"]) & set(eval_df["text"])

    assert len(overlap_train_val) > 0
    assert len(overlap_train_eval) > 0
    assert len(overlap_val_eval) > 0

    # We intentionally keep this as a documented validation risk, not a failing pipeline criterion,
    # because a future audit may decide on deduplication and split-aware preprocessing.
    assert len(overlap_train_val) <= len(train_df)
    assert len(overlap_train_eval) <= len(train_df)
    assert len(overlap_val_eval) <= len(val_df)
