import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.data import load_and_preprocess_data

def train_model():
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, r"E:\codealpha\models\model.pkl")
    print("✅ Model is trained and saved")

    # Evaluate model on test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2
    }

    # Return both model + metrics
    return model, metrics

if __name__ == "__main__":
    model, metrics = train_model()
    print("Metrics:", metrics)
