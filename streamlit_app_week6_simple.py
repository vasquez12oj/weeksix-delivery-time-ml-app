# This code creates a Streamlit web app that lets a user enter order details and get a predicted delivery time from a trained machine learning model.

# In simple terms:
#	the user enters inputs 
#	the app sends those inputs to the saved model 
#	the model predicts delivery time 
#	the app displays the result 

#Import Libraries
import streamlit as st                   # builds web app
import pandas as pd                      # creates a datafram for the model input
import pickle                            # loads the saved machine learning model


st.title("📦 Delivery Time Predictor")                              #Title for app

# Loads model
model_data=pickle.load(open("simple_week6_voting_model.pkl", "rb"))
model = model_data["model"]
features = model_data["features"]

st.write("Enter order details:")                                   # Displays instruction for user

purchase_dow = st.slider("Day of Week (0=Mon)", 0, 6, 1)           # User input widgets, information used for training
purchase_month = st.slider("Month", 1, 12, 1)
year = st.number_input("Year", value=2026)
product_size_cm3 = st.number_input("Product Size (cm³)", value=5000.0)
product_weight_g = st.number_input("Weight (g)", value=1000.0)
distance_km = st.number_input("Distance (km)", value=10.0)

if st.button("Predict"):                                           # Nothing happens until user clicks predict.  App runs predictio code below
    input_df = pd.DataFrame([[                   
        purchase_dow,
        purchase_month,
        year,
        product_size_cm3,
        product_weight_g,
        distance_km
    ]], columns=features)
                                                                   #input becomes a dataframe used for the scikit-learn models
    prediction = model.predict(input_df)[0]                         # sends input dataframe to the trained model
    st.success(f"Estimated delivery time: {round(prediction,2)} days") #rounds to 2 decimal places
