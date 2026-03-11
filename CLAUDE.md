# user instruction:
"After building or adding new features to the project or app, always run the tests and check the logs until every new functions and features work properly."

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```bash
python titanic_model.py
```

This runs the full end-to-end pipeline: EDA → feature engineering → imputation → encoding → cross-validation → hyperparameter tuning → prediction. Output is written to `submission.csv`.

## Dependencies

Key packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`

## Data Files

- `train.csv` — training set with `Transported` target column
- `test.csv` — test set (no target)
- `sample_submission.csv` — submission format reference
- `submission.csv` — generated output (after running the script)

## Architecture

`titanic_model.py` is a single-file script structured in 7 sequential sections:

1. **Data Loading & EDA** — loads CSVs, prints shape/dtypes/missing value counts
2. **Feature Engineering** (`engineer_features()`) — extracts `Group`, `GroupSize`, `IsAlone` from `PassengerId`; `Deck`, `CabinNum`, `Side` from `Cabin`; `FamilyName` from `Name`; spend aggregates (`TotalSpend`, `HasSpent`); `AgeGroup` bins
3. **Missing Value Imputation** — categorical columns filled with mode, numeric with median (computed on train+test combined to avoid leakage)
4. **Encoding** — boolean columns mapped to int; `HomePlanet`, `Destination`, `Deck`, `Side`, `AgeGroup` one-hot encoded using combined train+test to guarantee identical columns
5. **Cross-Validation** — 5-fold stratified CV across 5 models: LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM; best model selected by mean CV accuracy
6. **Hyperparameter Tuning** — `RandomizedSearchCV` (30 iterations) on the best model (LightGBM or XGBoost param grids defined)
7. **Final Prediction** — best estimator retrained on full training set, predictions written to `submission.csv` as boolean `Transported`

## Key Design Decisions

- Imputation stats are computed on the combined train+test split before encoding to prevent data leakage
- OHE is applied to the combined dataframe then split, ensuring train/test feature columns are always aligned
- `SPEND_COLS` NaNs are filled with 0 before CryoSleep zeroing (CryoSleep passengers cannot spend)
- The hyperparameter tuning only covers LightGBM and XGBoost; if another model wins CV, it falls back to the XGBoost search space
