import pickle
import streamlit as st
import pandas as pd

# Load the pickled model
with open('auto-mpg3.pkcls', 'rb') as file:
    model = pickle.load(file)

def predict_mpg(cylinders, displacement,horsepower, weight, acceleration):
    # Create a DataFrame with the input features
    data = pd.DataFrame({
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        #'model year': [model_year],
        #'origin': [origin]
    })
    
    # Use the loaded model to make predictions
    prediction = model.predict(data)
    
    return prediction[0]

def main():
    st.title("Auto-MPG Prediction")
    
    # Add input fields for user to enter values
    cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8])
    displacement = st.number_input("Displacement", min_value=0.0)
    horsepower = st.number_input("Horsepower", min_value=0.0)
    weight = st.number_input("Weight", min_value=0.0)
    acceleration = st.number_input("Acceleration", min_value=0.0)
    #model_year = st.selectbox("Model Year", [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82])
    #origin = st.selectbox("Origin", [1, 2, 3])
    
    # When the "Predict" button is clicked, make the prediction
    if st.button("Predict"):
        result = predict_mpg(cylinders, displacement,horsepower, weight, acceleration)
        st.success(f"The predicted MPG is: {result:.2f}")
if __name__ == '__main__':
    main()
    






