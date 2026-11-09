# src/predict.py
import joblib
import pandas as pd

from src.data import CATEGORICAL
from src.train import MODEL_PATH


def predict_new_car(car_details):
    """
    Predict a car's selling price (in lakhs).
    Accepts either:
      - dict of features
      - pandas DataFrame with one row
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: python -m src.train")
    model = joblib.load(MODEL_PATH)

    # If input is dict, convert to DataFrame
    if isinstance(car_details, dict):
        df = pd.DataFrame([car_details])
    elif isinstance(car_details, pd.DataFrame):
        df = car_details.copy()
    else:
        raise ValueError("Input must be a dict or a pandas DataFrame")

    # One-hot encode WITHOUT drop_first: with a single row, drop_first would delete
    # every category column and the model would ignore fuel, seller and transmission.
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL if c in df.columns])

    # Line up with the training columns (baseline categories simply drop out here)
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(df)[0]
    return round(float(prediction), 2)


if __name__ == "__main__":
    # Example test with dict
    new_car = {
        "Year": 2018,
        "Present_Price": 5.59,
        "Kms_Driven": 27000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
    }
    print("Predicted Selling Price:", predict_new_car(new_car))
