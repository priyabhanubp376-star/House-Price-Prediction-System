
from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folder
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download dataset
housing = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = housing.frame

# Save dataset
file_path = DATA_DIR / "california_housing.csv"
df.to_csv(file_path, index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("Saved at:", file_path)
print("\nFirst five records:")
print(df.head())
