# src/data.py
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths are worked out from this file's location, so the project runs from any folder or computer
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "car data.csv"

# The CSV uses these names; the app and predict.py use the names on the right.
# Renaming here keeps training and prediction consistent.
COLUMN_RENAMES = {"Driven_kms": "Kms_Driven", "Selling_type": "Seller_Type"}

CATEGORICAL = ["Fuel_Type", "Seller_Type", "Transmission"]


def load_data(filepath=DATA_PATH):
    """Load the car dataset, drop missing rows and standardise column names."""
    df = pd.read_csv(filepath).dropna()
    return df.rename(columns=COLUMN_RENAMES)


def load_and_preprocess_data(filepath=DATA_PATH):
    df = load_data(filepath)
    print("Dataset loaded successfully. Shape:", df.shape)

    # Features (X) and target (y)
    X = df.drop(["Car_Name", "Selling_Price"], axis=1)
    y = df["Selling_Price"]

    # One-hot encode the categorical columns (first category of each becomes the baseline)
    X = pd.get_dummies(X, columns=CATEGORICAL, drop_first=True)

    # 80/20 train-test split (fixed seed so results are repeatable)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_preprocess_data()
