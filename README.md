# 🚗 Used Car Price Prediction

![Actual vs predicted selling price on the test set](images/actual-vs-predicted.png)

Estimate a used car's selling price from its age, original price, mileage, fuel type, seller type and transmission, and find out which of those actually drive the price. A CodeAlpha data science project.

**Portfolio:** [syedahafsa135.github.io](https://syedahafsa135.github.io)

---

## Business question

A used-car dealer or private seller wants a fair asking price. **Which features of a car drive its resale value, and can we estimate a price from them reliably?**

## Data

- **Source:** used-car listings, in `data/car data.csv`
- **Size:** 301 cars, model years 2003–2018 (239 petrol, 60 diesel, 2 CNG)
- **Columns:** `Car_Name`, `Year`, `Selling_Price` (target, in lakhs), `Present_Price` (new price, in lakhs), `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`

## Approach

1. Load the data, drop rows with missing values, and standardise two column names (`Driven_kms` → `Kms_Driven`, `Selling_type` → `Seller_Type`) so training and the app use the same names
2. Drop `Car_Name` and one-hot encode `Fuel_Type`, `Seller_Type` and `Transmission`
3. Split 80% for training and 20% for testing (`random_state=42`)
4. Train a **Random Forest Regressor** (100 trees)
5. Evaluate on the **test set only** with MAE, RMSE and R²
6. Save the model (`models/model.pkl`) and serve predictions in a **Streamlit** app

## Key findings

<!-- TODO: fill in from the output of `python -m src.evaluate`. Use the printed numbers only. -->

| Metric (test set) | Value |
|---|---|
| MAE (lakhs) | _run `python -m src.evaluate`_ |
| RMSE (lakhs) | _run `python -m src.evaluate`_ |
| R² | _run `python -m src.evaluate`_ |

- **Strongest price driver:** _see the feature importance printed by `src.evaluate`_

![Feature importance](images/feature-importance.png)

![Selling price by fuel type](images/price-by-fuel-type.png)

## How to run

**Option A: Google Colab (nothing to install)**

```python
!git clone https://github.com/syedahafsa135/CodeAlpha_CarPricePrediction.git
%cd CodeAlpha_CarPricePrediction
!python -m src.train      # trains the model and prints test-set scores
!python -m src.evaluate   # prints scores + feature importance, saves charts to images/
```

**Option B: on your own computer**

```bash
git clone https://github.com/syedahafsa135/CodeAlpha_CarPricePrediction.git
cd CodeAlpha_CarPricePrediction
python -m venv venv
venv\Scripts\activate          # on Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
streamlit run app.py           # opens the prediction app in your browser
```

## Project structure

```
├── app.py              # Streamlit app: train, predict, about
├── data/car data.csv
├── models/model.pkl    # saved Random Forest model
├── images/             # charts created by src/evaluate.py
└── src/
    ├── data.py         # load, clean, encode and split
    ├── train.py        # train, save and score the model
    ├── evaluate.py     # test-set scores, feature importance, charts
    └── predict.py      # predict the price of a new car
```

## Tech stack

Python · pandas · NumPy · scikit-learn · joblib · Matplotlib · Streamlit
