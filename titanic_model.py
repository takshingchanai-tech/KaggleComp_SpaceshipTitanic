"""
Spaceship Titanic — Transported Prediction
Full end-to-end ML pipeline: EDA, feature engineering, training, evaluation, prediction
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────────────────────
# 1. Data Loading & EDA
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("1. DATA LOADING & EDA")
print("=" * 60)

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(f"\nTrain shape: {train.shape}")
print(f"Test  shape: {test.shape}")

# Print first 5 rows
print(train.head())
print(test.head())

print("\n--- dtypes ---")
print(train.dtypes)

print("\n--- Missing values (train) ---")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\n--- Missing values (test) ---")
print(test.isnull().sum()[test.isnull().sum() > 0])

for col in ["HomePlanet", "CryoSleep", "Destination", "Transported"]:
    if col in train.columns:
        print(f"\n{col} value counts:\n{train[col].value_counts(dropna=False)}")


# ─────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. FEATURE ENGINEERING")
print("=" * 60)

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- PassengerId: Group + GroupSize + IsAlone ---
    df["Group"] = df["PassengerId"].str.split("_").str[0].astype(int)
    group_sizes = df.groupby("Group")["PassengerId"].transform("count")
    df["GroupSize"] = group_sizes
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    # --- Cabin: Deck, CabinNum, Side ---
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["Deck"]     = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"]     = cabin_split[2]

    # --- Name: FamilyName ---
    df["FamilyName"] = df["Name"].str.split().str[-1]

    # --- Spend columns: fill NaN for non-CryoSleep first pass ---
    for col in SPEND_COLS:
        df[col] = df[col].fillna(0)

    # --- TotalSpend & HasSpent ---
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["HasSpent"]   = (df["TotalSpend"] > 0).astype(int)

    # --- CryoSleep consistency: if CryoSleep=True, zeroise spend ---
    cryo_true = df["CryoSleep"] == True
    for col in SPEND_COLS + ["TotalSpend"]:
        df.loc[cryo_true, col] = 0
    df.loc[cryo_true, "HasSpent"] = 0

    # --- Age bins ---
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 17, 35, 60, 200],
        labels=["child", "teen", "adult", "middle", "senior"]
    )

    return df


train = engineer_features(train)
test  = engineer_features(test)

print("Feature engineering complete.")
print(f"Train columns ({len(train.columns)}): {list(train.columns)}")


# ─────────────────────────────────────────────────────────────
# 3. Missing Value Imputation
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. MISSING VALUE IMPUTATION")
print("=" * 60)

CAT_COLS  = ["HomePlanet", "Destination", "Deck", "Side", "AgeGroup", "CryoSleep", "VIP"]
NUM_COLS  = ["Age", "CabinNum"] + SPEND_COLS + ["TotalSpend"]

# Combine for consistent imputation stats
combined = pd.concat([train, test], axis=0, ignore_index=True)

# Categorical → mode
for col in CAT_COLS:
    mode_val = combined[col].mode(dropna=True)[0]
    train[col] = train[col].fillna(mode_val)
    test[col]  = test[col].fillna(mode_val)
    print(f"  {col}: filled with mode='{mode_val}'")

# Numeric → median
for col in NUM_COLS:
    median_val = combined[col].median()
    train[col] = train[col].fillna(median_val)
    test[col]  = test[col].fillna(median_val)
    print(f"  {col}: filled with median={median_val:.2f}")


# ─────────────────────────────────────────────────────────────
# 4. Encoding
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. ENCODING")
print("=" * 60)

OHE_COLS  = ["HomePlanet", "Destination", "Deck", "Side", "AgeGroup"]
BOOL_COLS = ["CryoSleep", "VIP"]

# Boolean → int
for col in BOOL_COLS:
    train[col] = train[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    test[col]  = test[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)

# One-hot encode using combined to guarantee identical columns
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined_ohe = pd.get_dummies(combined, columns=OHE_COLS, drop_first=False, dtype=int)

train_len = len(train)
train_ohe = combined_ohe.iloc[:train_len].copy()
test_ohe  = combined_ohe.iloc[train_len:].copy()

print(f"After OHE — train: {train_ohe.shape}, test: {test_ohe.shape}")

# Drop columns not useful for modelling
DROP_COLS = ["PassengerId", "Name", "Cabin", "FamilyName", "Group", "Transported"]
feature_cols = [c for c in train_ohe.columns if c not in DROP_COLS]

X = train_ohe[feature_cols].astype(float)
y = train_ohe["Transported"].map({True: 1, False: 0, "True": 1, "False": 0}).astype(int)
X_test = test_ohe[feature_cols].astype(float)

print(f"Feature matrix X: {X.shape},  y: {y.shape},  X_test: {X_test.shape}")


# ─────────────────────────────────────────────────────────────
# 5. Model Training with 5-Fold CV
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. CROSS-VALIDATION (5-FOLD STRATIFIED)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    ),
}

cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_scores[name] = scores.mean()
    print(f"  {name:<22} CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

best_model_name = max(cv_scores, key=cv_scores.get)
print(f"\nBest model by CV accuracy: {best_model_name} ({cv_scores[best_model_name]:.4f})")


# ─────────────────────────────────────────────────────────────
# 6. Hyperparameter Tuning on Best Model
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. HYPERPARAMETER TUNING (RandomizedSearchCV)")
print("=" * 60)

if best_model_name == "LightGBM":
    param_dist = {
        "n_estimators":    [200, 300, 400, 500],
        "max_depth":       [4, 5, 6, 7, -1],
        "learning_rate":   [0.01, 0.03, 0.05, 0.07, 0.1],
        "subsample":       [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":[0.7, 0.8, 0.9, 1.0],
        "num_leaves":      [31, 63, 127],
        "min_child_samples":[10, 20, 30],
    }
    base_estimator = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
else:  # XGBoost (or fallback)
    param_dist = {
        "n_estimators":    [200, 300, 400, 500],
        "max_depth":       [3, 4, 5, 6, 7],
        "learning_rate":   [0.01, 0.03, 0.05, 0.07, 0.1],
        "subsample":       [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":[0.7, 0.8, 0.9, 1.0],
        "min_child_weight":[1, 3, 5],
        "gamma":           [0, 0.1, 0.2, 0.3],
    }
    base_estimator = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )

search = RandomizedSearchCV(
    base_estimator,
    param_distributions=param_dist,
    n_iter=30,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
search.fit(X, y)

print(f"Best params: {search.best_params_}")
print(f"Tuned CV accuracy: {search.best_score_:.4f}")

best_estimator = search.best_estimator_


# ─────────────────────────────────────────────────────────────
# 7. Final Prediction
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. FINAL PREDICTION")
print("=" * 60)

best_estimator.fit(X, y)
preds = best_estimator.predict(X_test)

test_ids = test["PassengerId"]
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": preds.astype(bool)
})

submission.to_csv("submission.csv", index=False)

print(f"submission.csv written — {len(submission)} rows")
print(submission.head(10))
print(f"\nTransported=True  count: {submission['Transported'].sum()}")
print(f"Transported=False count: {(~submission['Transported']).sum()}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
