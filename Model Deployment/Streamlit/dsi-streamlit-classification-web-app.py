# Import librariers

import streamlit as st
import pandas as pd 
import joblib 

# Load our model pipeline object

model = joblib.load(r"model.joblib")

# Add title and instructions

st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")

# Age input form
age = st.number_input(      
    label = "1\. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35
)


# Gender input form
gender = st.radio(
    label = "2\. Enter the customer's gender",
    options = ["M", "F"]
)


# Credit score input form
credit_score = st.number_input(      
    label = "3\. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500
)

# Submit inputs to model
if st.button("Submit for Prediction"):
    
    # Store our data in a dataframe for prediction
    
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    # Apply model pipeline to the input  data and extract probability prediction
    
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # Output prediction
    
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba: .0%}")
    
