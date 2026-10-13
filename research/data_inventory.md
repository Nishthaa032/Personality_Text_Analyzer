# Data inventory and data dictionary

## Verified dataset snapshot

The project currently contains three CSV files in `data/`:

- `train_set.csv` — 16,047 rows
- `val_set.csv` — 2,415 rows
- `eval_set.csv` — 2,415 rows

All three files share the same schema.

## Verified schema

| Column | Type | Description |
| --- | --- | --- |
| text | string | Reddit-style comment text |
| agreeableness | float | Big Five trait score, approximate 0–100 scale |
| openness | float | Big Five trait score, approximate 0–100 scale |
| conscientiousness | float | Big Five trait score, approximate 0–100 scale |
| extraversion | float | Big Five trait score, approximate 0–100 scale |
| neuroticism | float | Big Five trait score, approximate 0–100 scale |

## Label audit

The target variables are continuous personality scores, not categorical classes.

- Target type: continuous regression targets
- Scale: approximate 0–100 percentile-like values after external normalization
- Supported personality dimensions: agreeableness, openness, conscientiousness, extraversion, neuroticism
- Not present in provided files: participant IDs, user IDs, emotion labels, MBTI labels, location labels, demographic labels, train/validation/test labels beyond the split files themselves

## Data-quality checks

### Missing values

All required fields are non-null in the supplied CSVs.

### Duplicate rows

Exact duplicate rows are present, especially in the training split.

- train_set.csv: 2,768 duplicate rows
- val_set.csv: 131 duplicate rows
- eval_set.csv: 137 duplicate rows

### Duplicate texts across splits

Text-only overlap is present across splits, which is a leakage risk if not handled carefully.

- train ↔ val: 1,811 shared texts
- train ↔ eval: 1,806 shared texts
- val ↔ eval: 199 shared texts

This indicates that the provided split files are not deduplicated at the text level and should be treated with caution during model development.

## Licensing and provenance notes

The paper in `reserch paper/2021.socialnlp-1.12.pdf` describes a Reddit-based personality dataset (PANDORA) and explains that it was built from Reddit comments and flair-derived labels.

The current workspace does not include:

- a dataset license file
- a README describing permitted reuse
- a provenance manifest for the raw Reddit archive
- a consent/data-use statement specifically for the local CSVs

Therefore, the legal reuse status of the dataset cannot be guaranteed from the provided files alone.

## Leakage-risk assessment

The files are suitable for exploratory analysis and baseline modeling, but they require explicit safeguards before training:

1. deduplicate identical texts before modeling
2. audit cross-split overlap
3. preserve the test split until final evaluation
4. document any user-level leakage if user IDs are later supplied
5. avoid using any model-generated pseudo-labels as psychological ground truth

## Validation status

A validation test module is included under `tests/unit/test_data_validation.py` to assert the current schema, range constraints, null checks, and split-overlap documentation.

This chunk does not include any model training or emotion-model development. The emotion task remains unresolved because the GoEmotions dataset is not present in the provided files.
