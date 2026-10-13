# Personality model comparison summary

This report compares the two evaluated personality models on the verified Big Five dataset in this workspace.

## Scope

This project remains personality-only by design. The emotion task is intentionally deferred because the workspace does not contain the required GoEmotions dataset or emotion label schema.

## Models evaluated

1. TF-IDF + Ridge regression baseline
2. DistilBERT regression candidate

## Results

| Model | Split | Mean MAE | Mean RMSE | Mean Pearson |
|---|---|---:|---:|---:|
| TF-IDF + Ridge | Validation | 23.19 | 27.74 | 0.228 |
| TF-IDF + Ridge | Evaluation | 22.87 | 27.38 | 0.249 |
| DistilBERT | Validation | 23.85 | 27.30 | -0.024 |
| DistilBERT | Evaluation | 27.80 | 31.79 | -0.131 |

## Interpretation

For the verified personality dataset in this workspace, the TF-IDF + Ridge baseline remains the strongest model under current evidence. It achieves lower mean MAE and higher correlation than the DistilBERT regression candidate on both validation and evaluation splits.

The DistilBERT model is functional and validated as a candidate, but it does not outperform the baseline on the current evidence.

## Decision

The project continues with the TF-IDF + Ridge model as the deployed personality predictor, while emotion work remains blocked until the correct emotion dataset is supplied and validated.
