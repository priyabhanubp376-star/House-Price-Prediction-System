
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor

# Project directories
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
df = pd.read_csv(DATA_DIR / "california_housing.csv")

# Separate features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Create a fresh holdout split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=2026
)

print("Fresh holdout dataset created!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Use tuned hyperparameters
model = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="sqrt",
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining final model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nFINAL MODEL EVALUATION")
print("----------------------")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save final model
model_path = MODEL_DIR / "final_house_price_model.pkl"
joblib.dump(model, model_path)

print("\nFinal model saved successfully!")
print("Model location:", model_path)

# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.title("Actual vs Predicted House Values")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Residual analysis
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted House Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "residual_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nEvaluation charts saved successfully!")
