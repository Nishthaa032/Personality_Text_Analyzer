from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

TRAIT_COLUMNS = [
    "agreeableness",
    "openness",
    "conscientiousness",
    "extraversion",
    "neuroticism",
]
REQUIRED_COLUMNS = ["text", *TRAIT_COLUMNS]

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = RAW_DATA_DIR / "processed"


def load_raw_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].ne("")].copy()

    for col in TRAIT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[TRAIT_COLUMNS].notna().all(axis=1)].copy()
    return df.reset_index(drop=True)


def normalize_text_for_dedup(text: str) -> str:
    return " ".join(text.lower().split())


def count_normalized_duplicates(df: pd.DataFrame) -> int:
    normalized = df["text"].map(normalize_text_for_dedup)
    return int(normalized.duplicated().sum())


def remove_exact_and_normalized_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df["text"].map(normalize_text_for_dedup)
    df = df.assign(_normalized_text=normalized)
    deduped = df.drop_duplicates(subset=["_normalized_text"], keep="first").copy()
    deduped = deduped.drop(columns=["_normalized_text"]).reset_index(drop=True)
    return deduped


def remove_cross_split_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_texts = set(train_df["text"].tolist())
    val_without_train = val_df[~val_df["text"].isin(train_texts)].copy().reset_index(drop=True)
    eval_without_train = eval_df[~eval_df["text"].isin(train_texts)].copy().reset_index(drop=True)

    val_texts_after = set(val_without_train["text"].tolist())
    eval_without_val = eval_without_train[~eval_without_train["text"].isin(val_texts_after)].copy().reset_index(drop=True)
    val_without_eval = val_without_train[~val_without_train["text"].isin(eval_without_val["text"])].copy().reset_index(drop=True)

    assert not set(val_without_eval["text"]) & set(train_df["text"])
    assert not set(eval_without_val["text"]) & set(train_df["text"])
    assert not set(val_without_eval["text"]) & set(eval_without_val["text"])

    return train_df.reset_index(drop=True), val_without_eval.reset_index(drop=True), eval_without_val.reset_index(drop=True)


def build_processed_splits() -> Dict[str, object]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        "train": RAW_DATA_DIR / "train_set.csv",
        "val": RAW_DATA_DIR / "val_set.csv",
        "eval": RAW_DATA_DIR / "eval_set.csv",
    }

    raw_frames = {name: load_raw_split(path) for name, path in raw_paths.items()}
    raw_report = {
        name: {
            "rows": len(df),
            "normalized_duplicate_texts": count_normalized_duplicates(df),
        }
        for name, df in raw_frames.items()
    }

    deduped = {name: remove_exact_and_normalized_duplicates(df) for name, df in raw_frames.items()}
    dedup_report = {name: {"rows_after_dedup": len(df)} for name, df in deduped.items()}

    train_df, val_df, eval_df = deduped["train"], deduped["val"], deduped["eval"]
    train_df, val_df, eval_df = remove_cross_split_overlap(train_df, val_df, eval_df)

    processed_data = {"train": train_df, "val": val_df, "eval": eval_df}
    for split_name, frame in processed_data.items():
        output_path = PROCESSED_DIR / f"{split_name}_processed.csv"
        frame.to_csv(output_path, index=False)

    report = {
        "source_root": str(RAW_DATA_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "raw_counts": raw_report,
        "deduplicated_counts": dedup_report,
        "processed_split_counts": {name: len(df) for name, df in processed_data.items()},
        "final_overlap_free": {
            "train_val_overlap": int(len(set(processed_data["train"]["text"]) & set(processed_data["val"]["text"]))),
            "train_eval_overlap": int(len(set(processed_data["train"]["text"]) & set(processed_data["eval"]["text"]))),
            "val_eval_overlap": int(len(set(processed_data["val"]["text"]) & set(processed_data["eval"]["text"]))),
        },
        "trait_columns": TRAIT_COLUMNS,
    }

    manifest_path = PROCESSED_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    build_processed_splits()
    print(f"Processed split files written to {PROCESSED_DIR}")
