
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "california_housing.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Original shape:", df.shape)

    # Separate features and target
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Save training and testing data
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("\nData preprocessing completed!")

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("Training target:", y_train.shape)
    print("Testing target:", y_test.shape)


if __name__ == "__main__":
    prepare_data()
