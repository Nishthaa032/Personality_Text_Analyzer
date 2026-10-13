# Personality Text Analyzer

A research-oriented personality prediction project built from the verified dataset available in this workspace. The project stays strictly within the personality-only scope because the required emotion dataset and labels are not present.

## Project scope and constraints

This project is intentionally personality-only:

- Verified dataset: Reddit-derived Big Five personality scores
- Target traits: agreeableness, openness, conscientiousness, extraversion, neuroticism
- Emotion modeling: deferred until the exact GoEmotions dataset and schema are provided and validated

The dataset in this workspace does not include emotion labels, emotion annotations, or a verified emotion training set.

## Architecture

- Data validation and dataset inventory: `src/data/data_inventory.py`
- Leakage-safe preprocessing and split generation: `src/preprocessing/build_processed_splits.py`
- TF-IDF + Ridge baseline model: `src/personality/tfidf_regression_baseline.py`
- DistilBERT regression candidate: `src/personality/distilbert_regression.py`
- Prediction API: `src/api/app.py`
- Personality interpretation summary: `src/personality/personality_summary.py`
- Frontend UI: `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`

## Model comparison

The current verified comparison between the two tested models is summarized in [research/model_comparison_summary.md](research/model_comparison_summary.md) and exported to [research/model_comparison_summary.json](research/model_comparison_summary.json).

### Summary table

| Model | Split | Mean MAE | Mean RMSE | Mean Pearson |
|---|---|---:|---:|---:|
| TF-IDF + Ridge | Validation | 23.19 | 27.74 | 0.228 |
| TF-IDF + Ridge | Evaluation | 22.87 | 27.38 | 0.249 |
| DistilBERT | Validation | 23.85 | 27.30 | -0.024 |
| DistilBERT | Evaluation | 27.80 | 31.79 | -0.131 |

Conclusion: the TF-IDF + Ridge baseline is the strongest model currently supported by evidence in this workspace.

## Personality summary feature

The browser app also returns a plain-language personality summary below the
predicted scores. It is generated locally from the five model outputs and does
not require a second model, external API, or additional credentials. It describes
the highest and lowest predicted dimensions and includes a non-clinical
interpretation disclaimer. The API endpoint is:

```bash
curl -X POST http://127.0.0.1:8011/summarize \
  -H "Content-Type: application/json" \
  -d '{"text":"I enjoy planning trips with friends and trying new activities."}'
```

## Quick start

### 1) Start the API

```bash
cd /Users/harshsmac/Desktop/Text_to_personality
uvicorn src.api.app:app --host 127.0.0.1 --port 8011
```

### 2) Serve the frontend

```bash
cd /Users/harshsmac/Desktop/Text_to_personality
python -m http.server 8081 --directory frontend
```

### 3) Open the app

Open the browser at:

```text
http://127.0.0.1:8081/
```

### 4) Test API directly

```bash
curl -X POST http://127.0.0.1:8011/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I enjoy spending time with friends and making plans."}'
```

## Model artifacts

- Baseline model: `models/personality_tfidf_ridge.joblib`
- Baseline vectorizer: `models/personality_tfidf_vectorizer.joblib`
- DistilBERT model directory: `models/personality_distilbert/distilbert_personality_regression`

## Validation and testing

Run the project unit tests with:

```bash
cd /Users/harshsmac/Desktop/Text_to_personality
pytest tests/unit -q
```

Current verified result: 18 passed.

## Data and research notes

- Processed split data: `data/processed/train_processed.csv`, `val_processed.csv`, `eval_processed.csv`
- Research notes: `research/data_inventory.md`
- Baseline metrics: `research/personality_tfidf_baseline.json`
- DistilBERT metrics: `research/personality_distilbert_regression.json`

## Limitations

- No verified emotion dataset is present in this project workspace.
- The model is currently a personality predictor only.
- The DistilBERT run is a candidate implementation and not the best-performing current model on the available evidence.
