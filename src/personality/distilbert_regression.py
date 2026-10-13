from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TRAITS = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models" / "personality_distilbert"
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


def compute_target_stats(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_df[TRAITS].mean(axis=0).to_numpy(dtype=np.float32)
    std = train_df[TRAITS].std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return mean, std


def inverse_transform_targets(y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return y_scaled * std + mean


class PersonalityTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: AutoTokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        return item


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    return pd.read_csv(path)


def build_dataloader(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool = False,
    target_mean: Optional[np.ndarray] = None,
    target_std: Optional[np.ndarray] = None,
):
    texts = df["text"].astype(str).tolist()
    labels = df[TRAITS].to_numpy(dtype=np.float32)
    if target_mean is not None and target_std is not None:
        labels = (labels - target_mean) / target_std
    dataset = PersonalityTextDataset(texts, labels, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    pearson = np.array([pearson_r(y_true[:, idx], y_pred[:, idx]) for idx in range(len(TRAITS))])
    return {
        "mae_by_trait": {trait: float(score) for trait, score in zip(TRAITS, mae)},
        "rmse_by_trait": {trait: float(score) for trait, score in zip(TRAITS, rmse)},
        "pearson_by_trait": {trait: float(score) for trait, score in zip(TRAITS, pearson)},
        "mean_mae": float(np.mean(mae)),
        "mean_rmse": float(np.mean(rmse)),
        "mean_pearson": float(np.mean(pearson)),
    }


def evaluate_model(model: AutoModelForSequenceClassification, data_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    y_pred = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_pred


def train_and_evaluate(
    epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    max_length: int = 128,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    target_normalization: bool = True,
) -> Dict[str, object]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train").reset_index(drop=True)
    val_df = load_split("val").reset_index(drop=True)
    eval_df = load_split("eval").reset_index(drop=True)

    if max_train_samples is not None:
        train_df = train_df.head(max_train_samples).reset_index(drop=True)
    if max_val_samples is not None:
        val_df = val_df.head(max_val_samples).reset_index(drop=True)
    if max_eval_samples is not None:
        eval_df = eval_df.head(max_eval_samples).reset_index(drop=True)

    target_mean = None
    target_std = None
    if target_normalization:
        target_mean, target_std = compute_target_stats(train_df)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(TRAITS),
        problem_type="regression",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = build_dataloader(
        train_df,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_loader = build_dataloader(
        val_df,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        target_mean=target_mean,
        target_std=target_std,
    )
    eval_loader = build_dataloader(
        eval_df,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        target_mean=target_mean,
        target_std=target_std,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        val_true_scaled, val_pred_scaled = evaluate_model(model, val_loader, device)
        if target_mean is not None and target_std is not None:
            val_true = inverse_transform_targets(val_true_scaled, target_mean, target_std)
            val_pred = inverse_transform_targets(val_pred_scaled, target_mean, target_std)
        else:
            val_true = val_true_scaled
            val_pred = val_pred_scaled

        val_loss = float(loss_fn(torch.tensor(val_pred, dtype=torch.float32).to(device), torch.tensor(val_true, dtype=torch.float32).to(device)))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_true_scaled, val_pred_scaled = evaluate_model(model, val_loader, device)
    eval_true_scaled, eval_pred_scaled = evaluate_model(model, eval_loader, device)

    if target_mean is not None and target_std is not None:
        val_true = inverse_transform_targets(val_true_scaled, target_mean, target_std)
        val_pred = inverse_transform_targets(val_pred_scaled, target_mean, target_std)
        eval_true = inverse_transform_targets(eval_true_scaled, target_mean, target_std)
        eval_pred = inverse_transform_targets(eval_pred_scaled, target_mean, target_std)
    else:
        val_true = val_true_scaled
        val_pred = val_pred_scaled
        eval_true = eval_true_scaled
        eval_pred = eval_pred_scaled

    metric_rows = []
    for split_name, y_true, y_pred in [("val", val_true, val_pred), ("eval", eval_true, eval_pred)]:
        metrics = compute_regression_metrics(y_true, y_pred)
        metric_rows.append({"split": split_name, **metrics})

    model_path = MODEL_DIR / "distilbert_personality_regression"
    tokenizer.save_pretrained(str(model_path))
    model.save_pretrained(str(model_path))

    report = {
        "description": "DistilBERT fine-tuning regression baseline for five Big Five personality traits",
        "model_name": "distilbert-base-uncased",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_length": max_length,
        "target_normalization": target_normalization,
        "trait_columns": TRAITS,
        "metrics": metric_rows,
        "model_dir": str(model_path),
    }
    report_path = RESULTS_DIR / "personality_distilbert_regression.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = train_and_evaluate()
    print(json.dumps(report["metrics"], indent=2))
