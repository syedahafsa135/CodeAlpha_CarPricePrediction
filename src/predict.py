import joblib
import pandas as pd
import os

# Load model
model_path = os.path.join("models", "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")

model = joblib.load(model_path)

def predict_new_car(car_details):
    """
    Predict car price.
    Accepts either:
      - dict of features
      - pandas DataFrame with one row
    """
    # If input is dict, convert to DataFrame
    if isinstance(car_details, dict):
        df = pd.DataFrame([car_details])
    elif isinstance(car_details, pd.DataFrame):
        df = car_details.copy()
    else:
        raise ValueError("Input must be a dict or a pandas DataFrame")

    # One-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Ensure same columns as training
    model_features = model.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    # Predict
    prediction = model.predict(df)[0]
    return round(prediction, 2)


if __name__ == "__main__":
    # Example test with dict
    new_car = {
        "Year": 2018,
        "Present_Price": 5.59,
        "Kms_Driven": 27000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0
    }
    price = predict_new_car(new_car)
    print("Predicted Selling Price:", price)
