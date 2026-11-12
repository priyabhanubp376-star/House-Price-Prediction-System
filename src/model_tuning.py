
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

# Project directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

# Load training data only
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

print("Training data loaded successfully!")
print("Training shape:", X_train.shape)

# Base Random Forest model
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=1
)

# Hyperparameter combinations
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 15, 20, 25],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 1.0]
}

# 5-fold cross validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Randomized Search
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nStarting Hyperparameter Tuning...")

search.fit(X_train, y_train)

# Display results
print("\nTUNING COMPLETED!")

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross-Validation RMSE:")
print(round(-search.best_score_, 4))

print("\nBest Model:")
print(search.best_estimator_)

import joblib

# Model folder
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Get best tuned model
best_model = search.best_estimator_

# Save tuned model
model_path = MODEL_DIR / "tuned_house_price_model.pkl"

joblib.dump(best_model, model_path)

print("\nTuned model saved successfully!")
print("Model location:", model_path)