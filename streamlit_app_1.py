import streamlit as st
import requests
import json

import pandas as pd



# Function to call the deployed model's prediction endpoint
def predict(input_data,):
    url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(input_data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}
    




st.title("End to End Customer Satisfaction Pipeline with ZenML")

payment_sequential = st.sidebar.slider("Payment Sequential")
payment_installments = st.sidebar.slider("Payment Installments")
payment_value = st.number_input("Payment Value")
price = st.number_input("Price")
freight_value = st.number_input("freight_value")
product_name_length = st.number_input("Product name length")
product_description_length = st.number_input("Product Description length")
product_photos_qty = st.number_input("Product photos Quantity ")
product_weight_g = st.number_input("Product weight measured in grams")
product_length_cm = st.number_input("Product length (CMs)")
product_height_cm = st.number_input("Product height (CMs)")
product_width_cm = st.number_input("Product width (CMs)")

df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
)

if st.button("Predict"):
    prediction = predict(df)

    if "error" in prediction:
        st.error(f"Error: {prediction['error']}")
    else:
        st.success(f"Prediction: {prediction}")

