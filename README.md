# Spaceship Titanic — Kaggle Competition

Kaggle competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

**Goal:** Predict which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python titanic_model.py
```

Produces `submission.csv` ready for upload to Kaggle.

## Pipeline

1. EDA — shape, dtypes, missing value summary
2. Feature engineering — group info from `PassengerId`, cabin components, spend aggregates, age bins
3. Imputation — mode for categoricals, median for numerics
4. Encoding — boolean columns to int, one-hot encoding for categoricals
5. Cross-validation — 5-fold stratified CV across Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
6. Hyperparameter tuning — `RandomizedSearchCV` on the best CV model
7. Prediction — retrain on full data, write `submission.csv`

## Data

Download from the competition page and place in the project root:
- `train.csv`
- `test.csv`
- `sample_submission.csv`
