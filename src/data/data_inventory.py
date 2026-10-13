from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

TRAIT_COLUMNS = [
    "agreeableness",
    "openness",
    "conscientiousness",
    "extraversion",
    "neuroticism",
]


def load_split_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_traits = {col: df[col].astype(float) for col in TRAIT_COLUMNS if col in df.columns}
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "text_column": "text" if "text" in df.columns else None,
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()) if "text" in df.columns else 0,
        "unique_text_count": int(df["text"].nunique()) if "text" in df.columns else 0,
        "trait_summary": {
            key: {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std(ddof=0)),
            }
            for key, series in numeric_traits.items()
        },
    }
    return summary


def cross_split_overlap(split_frames: Iterable[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    frames = list(split_frames)
    names = [f"split_{idx + 1}" for idx in range(len(frames))]
    outputs: Dict[str, Dict[str, Any]] = {}
    for i in range(len(frames)):
        left = frames[i]
        for j in range(i + 1, len(frames)):
            right = frames[j]
            overlap = set(left["text"]) & set(right["text"])
            key = f"{names[i]}__{names[j]}"
            outputs[key] = {
                "overlap_text_count": len(overlap),
                "left_rows": int(len(left)),
                "right_rows": int(len(right)),
                "ratio_left": round(len(overlap) / len(left), 6) if len(left) else 0.0,
                "ratio_right": round(len(overlap) / len(right), 6) if len(right) else 0.0,
                "examples": sorted(list(overlap))[:5],
            }
    return outputs


def build_inventory(data_dir: str | Path) -> Dict[str, Any]:
    root = Path(data_dir)
    split_paths = [root / "train_set.csv", root / "val_set.csv", root / "eval_set.csv"]
    frames = [load_split_csv(path) for path in split_paths]
    inventory = {
        "dataset_root": str(root),
        "verified_target_columns": TRAIT_COLUMNS,
        "target_type": "continuous_big_five_scores",
        "score_scale": "0-100 approximated percentile-like trait score",
        "split_files": [str(path.name) for path in split_paths],
        "split_summaries": {
            path.name: summarize_dataframe(df)
            for path, df in zip(split_paths, frames)
        },
        "cross_split_overlap": cross_split_overlap(frames),
    }
    return inventory


def write_inventory(data_dir: str | Path, output_path: str | Path) -> Dict[str, Any]:
    inventory = build_inventory(data_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    output_path = Path(__file__).resolve().parents[2] / "research" / "data_inventory.json"
    write_inventory(data_dir, output_path)
    print(f"Inventory written to {output_path}")
