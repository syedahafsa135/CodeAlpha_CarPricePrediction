# src/evaluate.py
# Scores the saved model on the held-out TEST set only (the cars it never saw in training),
# prints the metrics and feature importance, and saves three charts to images/.
#
# Run from the repo folder:  python -m src.evaluate
import joblib
import matplotlib

matplotlib.use("Agg")  # save charts to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import ROOT, load_and_preprocess_data, load_data
from src.train import MODEL_PATH

IMAGES = ROOT / "images"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run: python -m src.train")

    model = joblib.load(MODEL_PATH)
    _, X_test, _, y_test = load_and_preprocess_data()
    X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("\n📊 Test-set results (held-out cars only, prices in lakhs):")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {np.sqrt(mse):.2f}")
    print(f"  R²:   {r2_score(y_test, y_pred):.3f}")

    importance = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
    print("\n📌 Feature importance:")
    for name, value in importance.sort_values(ascending=False).items():
        print(f"  {name}: {value:.3f}")

    IMAGES.mkdir(exist_ok=True)

    # 1. Actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, color="#0E7C86", alpha=0.8)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="#9AA5B1", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual selling price (lakhs)")
    ax.set_ylabel("Predicted selling price (lakhs)")
    ax.set_title("Actual vs predicted price (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "actual-vs-predicted.png", dpi=150)

    # 2. Feature importance
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance.index, importance.values, color="#0E7C86")
    ax.set_xlabel("Importance")
    ax.set_title("What drives the predicted price?")
    fig.tight_layout()
    fig.savefig(IMAGES / "feature-importance.png", dpi=150)

    # 3. Selling price by fuel type
    df = load_data()
    groups = [(name, g["Selling_Price"].values) for name, g in df.groupby("Fuel_Type")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([v for _, v in groups])
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{name}\n(n={len(v)})" for name, v in groups])
    ax.set_ylabel("Selling price (lakhs)")
    ax.set_title("Selling price by fuel type")
    fig.tight_layout()
    fig.savefig(IMAGES / "price-by-fuel-type.png", dpi=150)

    print(f"\n🖼️  Charts saved in {IMAGES}")


if __name__ == "__main__":
    main()
