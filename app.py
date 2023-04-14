import gradio as gr
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer
import pandas as pd

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

data = load_breast_cancer()
# Load the pre-trained logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to use the model for predicting diagnosis
def predict_diagnosis(
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, 
    fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, 
    smoothness_se, compactness_se, concavity_se, concave_points_se, 
    symmetry_se, fractal_dimension_se, radius_worst, texture_worst, 
    perimeter_worst, area_worst, smoothness_worst, compactness_worst, 
    concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst):

    features = np.array([
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, 
        fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, 
        smoothness_se, compactness_se, concavity_se, concave_points_se, 
        symmetry_se, fractal_dimension_se, radius_worst, texture_worst, 
        perimeter_worst, area_worst, smoothness_worst, compactness_worst, 
        concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]).reshape(1, -1)
    scaled_features = scaler.transform(features)
    diagnosis = model.predict(scaled_features)
    
    return 'Malignant' if diagnosis[0] == 'M' else 'Benign'

# Define Gradio input components for all the features
input_features = [
    gr.inputs.Number(label=feature_name) for feature_name in data.feature_names
]

# Define Gradio output component for diagnosis
output_diagnosis = gr.outputs.Textbox(label="Diagnosis")

# Create a Gradio user interface with the inputs, outputs, and the predict_diagnosis function
iface = gr.Interface(predict_diagnosis, input_features, output_diagnosis, title="Breast Cancer Diagnosis", description="Enter the features of the tumor to predict the diagnosis.")

# Launch the Gradio interface
iface.launch()


