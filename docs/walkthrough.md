# Walkthrough: Ensemble Repo Scaffold

## What was built

A complete repo skeleton for 4 team members to work in parallel on binary classification (stroke prediction), then integrate into an ensemble.

## Files created/modified

### Shared utilities (`src/`)

| File | Purpose |
|---|---|
| [data_loader.py](file:///home/mmw/Documents/MIA/ml/AdM_TP/src/data_loader.py) | Load dataset + fixed-seed train/test split. Points to `healthcare-dataset-stroke-data.csv`, target = `stroke` |
| [evaluation.py](file:///home/mmw/Documents/MIA/ml/AdM_TP/src/evaluation.py) | Metrics (accuracy, precision, recall, F1, AUC), confusion matrix, ROC curve, comparison table |
| [model_registry.py](file:///home/mmw/Documents/MIA/ml/AdM_TP/src/model_registry.py) | Save/load models via joblib + params.yaml metadata |

### Notebooks (`notebooks/`)

| Notebook | Owner |
|---|---|
| [00_eda.ipynb](file:///home/mmw/Documents/MIA/ml/AdM_TP/notebooks/00_eda.ipynb) | Shared EDA |
| [01_model_A.ipynb](file:///home/mmw/Documents/MIA/ml/AdM_TP/notebooks/01_model_A.ipynb) | Member 1 |
| [02_model_B.ipynb](file:///home/mmw/Documents/MIA/ml/AdM_TP/notebooks/02_model_B.ipynb) | Member 2 |
| [03_model_C.ipynb](file:///home/mmw/Documents/MIA/ml/AdM_TP/notebooks/03_model_C.ipynb) | Member 3 |
| [04_model_D.ipynb](file:///home/mmw/Documents/MIA/ml/AdM_TP/notebooks/04_model_D.ipynb) | Member 4 |
| [05_ensemble.ipynb](file:///home/mmw/Documents/MIA/ml/AdM_TP/notebooks/05_ensemble.ipynb) | Integration: hard/soft/weighted voting + stacking |

### Config

| File | Changes |
|---|---|
| [pyproject.toml](file:///home/mmw/Documents/MIA/ml/AdM_TP/pyproject.toml) | Added sklearn, pandas, numpy, matplotlib, seaborn, jupyter, etc. Optional: xgboost, lightgbm, optuna |
| [.gitignore](file:///home/mmw/Documents/MIA/ml/AdM_TP/.gitignore) | Ignores data/, model binaries, notebook checkpoints |
| [README.md](file:///home/mmw/Documents/MIA/ml/AdM_TP/README.md) | Setup instructions, structure, workflow |

## Dataset

The stroke dataset (`healthcare-dataset-stroke-data.csv`) was already in `data/`. 5,111 rows, 12 columns, binary target `stroke`.

## Next steps

1. Run `uv sync` to install dependencies.
2. Each member opens their notebook, picks a model, and starts working.
3. After all 4 models are saved, run `05_ensemble.ipynb`.
