
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

# Project directories
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load training features
X_train = pd.read_csv(DATA_DIR / "X_train.csv")

# Load tuned model
model_path = MODEL_DIR / "tuned_house_price_model.pkl"
model = joblib.load(model_path)

print("Tuned model loaded successfully!")

# Extract feature importance
importance = model.feature_importances_

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE:")
print(importance_df.to_string(index=False))

# Create visualization
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("House Price Prediction - Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

# Save figure
figure_path = FIGURE_DIR / "feature_importance.png"

plt.savefig(figure_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nFeature importance chart saved successfully!")
print("Saved at:", figure_path)
