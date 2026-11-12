
import pandas as pd

from pathlib import Path
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Project directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"


# Load training data only
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

# Define models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# 5-fold cross validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():

    print(f"\nValidating {name}...")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        n_jobs=1
    )

    results.append({
        "Model": name,
        "Mean R2": scores["test_r2"].mean(),
        "Mean MAE": -scores["test_mae"].mean(),
        "Mean RMSE": -scores["test_rmse"].mean()
    })

results_df = pd.DataFrame(results)

print("\n5-FOLD CROSS VALIDATION RESULTS")
print(results_df.to_string(index=False))
