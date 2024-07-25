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


input_data = {}
input_data["payment_sequential"] = st.text_input("payment_sequential")
input_data["payment_installments"] = st.text_input("payment_installments")
input_data["payment_value"] = st.text_input("payment_value")
input_data["price"] = st.text_input("price")
input_data["freight_value"] = st.text_input("freight_value")
input_data["product_name_length"] = st.text_input("product_name_length")
input_data["product_description_length"] = st.text_input("product_description_length")
input_data["product_photos_qty"] = st.text_input("product_photos_qty")
input_data["product_weight_g"] = st.text_input("product_weight_g")
input_data["product_length_cm"] = st.text_input("product_length_cm")
input_data["product_height_cm"] = st.text_input("product_height_cm")
input_data["product_width_cm"] = st.text_input("product_width_cm")




if st.button("Predict"):

    input_dict = {key: [value] for key, value in input_data.items()}
    prediction = predict(input_dict)

    if "error" in prediction:
        st.error(f"Error: {prediction['error']}")
    else:
        st.success(f"Prediction: {prediction}")

