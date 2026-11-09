# src/train.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import ROOT, load_and_preprocess_data

MODEL_PATH = ROOT / "models" / "model.pkl"


def train_model():
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("✅ Model trained and saved to", MODEL_PATH)

    # Evaluate on the held-out test set only
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(y_test, y_pred),
    }
    return model, metrics


if __name__ == "__main__":
    _, metrics = train_model()
    print("\n📊 Test-set results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
