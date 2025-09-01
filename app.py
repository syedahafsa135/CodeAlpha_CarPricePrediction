# app.py
import streamlit as st
import pandas as pd
import os

from src.train import train_model
from src.predict import predict_new_car

# App title
st.title("🚗 Car Price Prediction App")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Train Model", "Predict Price", "About"])

# --- Train Model Page ---
if page == "Train Model":
    st.header("🔧 Train the Model")

    data_path = "data/car data.csv"

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}")
    else:
        if st.button("Train Model"):
            with st.spinner("Training the model... Please wait ⏳"):
                model, metrics = train_model()
            st.success("✅ Model trained successfully!")

            # Show metrics
            st.subheader("📊 Model Performance")
            for k, v in metrics.items():
                st.write(f"**{k}:** {v:.4f}")

# --- Predict Price Page ---
elif page == "Predict Price":
    st.header("💡 Predict Car Price")

    # Collect user inputs
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, value=20000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.number_input("Number of Previous Owners", min_value=0, max_value=3, value=0)

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "Year": year,
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type,
            "Transmission": transmission,
            "Owner": owner
        }])

        try:
            price = predict_new_car(input_data)
            st.success(f"💰 Predicted Selling Price: **{price:.2f} lakhs**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- About Page ---
elif page == "About":
    st.header("ℹ️ About this Project")
    st.markdown("""
    This Car Price Prediction App uses **Machine Learning** to estimate the 
    resale value of a car based on its features.  

    **Features:**
    - Train a regression model on the provided dataset  
    - Predict car prices based on user input  
    - Interactive and easy-to-use interface powered by Streamlit 🚀  
    """)

