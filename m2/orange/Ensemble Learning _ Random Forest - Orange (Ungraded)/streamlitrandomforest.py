import pickle
import Orange
import streamlit as st
import pandas as pd
import numpy as np 

# Load the pickled model
model = pickle.load(open('randomforest.pkcls', 'rb'))

def predict_DEATH_EVENT(creatinine_phosphokinase,ejection_fraction,platelets,serum_creatinine,serum_sodium,age,time):
    # Create a DataFrame with the input features
    data = pd.DataFrame({
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'ejection_fraction': [ejection_fraction],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'age': [age],
        'time': [time]
    })
    
    # Use the loaded model to make predictions
    prediction = model.predict(data)
    
    return prediction[0]

def main():
    st.title("Death Event Prediction")
    
    # Add input fields for user to enter values
    creatinine_phosphokinase = st.number_input("creatinine_phosphokinase", min_value=0.0)
    ejection_fraction = st.number_input("ejection_fraction", min_value=0.0)
    platelets = st.number_input("platelets", min_value=0.0)
    serum_creatinine = st.number_input("serum_creatinine", min_value=0.0)
    serum_sodium = st.number_input("serum_sodium", min_value=100)
    age = st.number_input("age", min_value=0.0)
    time = st.number_input("time", min_value=0.0)
    
    # When the "Predict" button is clicked, make the prediction
    if st.button("Predict"):
        result = predict_DEATH_EVENT(creatinine_phosphokinase,ejection_fraction,platelets,serum_creatinine,serum_sodium,age,time)
        st.success(f"The predicted DEATH_EVENT is: {result}")
if __name__ == '__main__':
    main()
