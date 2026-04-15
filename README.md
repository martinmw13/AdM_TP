# AdM TP - Ensemble Binary Classification

Team project: 4 individual models combined into an ensemble with multiple voting strategies.

## Setup

```bash
uv sync
# Optional: install xgboost/lightgbm
uv sync --extra boost
```

## Project Structure

```
├── data/raw/              # Place the clean dataset here (not committed)
├── src/                   # Shared utilities (data loading, evaluation, model registry)
├── notebooks/
│   ├── 00_eda.ipynb       # Exploratory Data Analysis
│   ├── 01_model_A.ipynb   # Member 1
│   ├── 02_model_B.ipynb   # Member 2
│   ├── 03_model_C.ipynb   # Member 3
│   ├── 04_model_D.ipynb   # Member 4
│   └── 05_ensemble.ipynb  # Ensemble integration & comparison
├── artifacts/             # Saved models + hyperparameters (YAML tracked, joblib ignored)
└── reports/               # Final consolidated report
```

## Workflow

1. **All members**: Place the dataset in `data/raw/` and update `src/data_loader.py` with the correct filename and target column.
2. **Each member**: Work in your assigned notebook (`01`-`04`). Use the shared utilities from `src/` for data loading, evaluation, and model saving.
3. **Integration**: Once all 4 models are saved to `artifacts/`, run `05_ensemble.ipynb` to build and compare ensemble strategies.

### Key rule: everyone uses `src.data_loader.get_train_test_split()`

This guarantees identical train/test splits across all models (fixed seed = 13).

## Ensemble Strategies

The ensemble notebook (`05_ensemble.ipynb`) evaluates:

| Strategy        | Description                                     |
| --------------- | ----------------------------------------------- |
| Hard Voting     | Majority vote on predicted labels               |
| Soft Voting     | Average predicted probabilities                 |
| Weighted Voting | Soft voting with weights from CV scores         |
| Stacking        | Meta-learner (LogReg) on base model predictions |
