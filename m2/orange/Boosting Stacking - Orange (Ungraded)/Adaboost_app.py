import gradio as gr
import numpy as np
import pickle

# Load the trained AdaBoostClassifier model from the pickle file.
with open('Adaboost.pkcls', 'rb') as file:
    ADB = pickle.load(file)

# Rest of your code remains the same from here.

# Function to predict liver disease using the loaded model.
def predict_liver_disease(Gender, Total_Bilirubin, Alkaline_Phosphotase, Aspartate_Aminotransferase, Albumin, Albumin_and_Globulin_Ratio):
    Gender = labelencoder.transform([Gender])[0]
    input_data = np.array([[Gender, Total_Bilirubin, Alkaline_Phosphotase, Aspartate_Aminotransferase, Albumin, Albumin_and_Globulin_Ratio]])
    scaled_input_data = scaler.transform(input_data)
    prediction = ADB.predict(scaled_input_data)[0]
    return "Positive" if prediction == 1 else "Negative"

# Define the input fields for the Gradio interface
inputs = [
    gr.inputs.Dropdown(["Male", "Female"], label="Gender"),
    gr.inputs.Number(label="Total Bilirubin"),
    gr.inputs.Number(label="Alkaline Phosphotase"),
    gr.inputs.Number(label="Aspartate Aminotransferase"),
    gr.inputs.Number(label="Albumin"),
    gr.inputs.Number(label="Albumin and Globulin Ratio")
]

# Define the output field for the Gradio interface
output = gr.outputs.Label()

# Create the Gradio interface with the predict_liver_disease function as the underlying model
gr_interface = gr.Interface(fn=predict_liver_disease, inputs=inputs, outputs=output)

# Launch the Gradio interface
gr_interface.launch()