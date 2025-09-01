import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Paths
data_path = os.path.join("data", "car data.csv")
model_path = os.path.join("models", "model.pkl")

# Load dataset
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please check your data folder.")

df = pd.read_csv(data_path)

# Preprocessing (match train.py exactly!)
X = df.drop(["Car_Name", "Selling_Price"], axis=1)
y = df["Selling_Price"]

# One-hot encode categorical columns (same as training)
X = pd.get_dummies(X, drop_first=True)

# Load trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")

model = joblib.load(model_path)

# Ensure feature alignment
model_features = model.feature_names_in_
X = X.reindex(columns=model_features, fill_value=0)

# Predict
y_pred = model.predict(X)

# Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
