#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : luis-eduardo@dsv.su.se
# Created Date: 2020/06/30
# =============================================================================
"""
Course: Data Mining for Computer and Systems Sciences

Homework 3.3: Model Deployment

Creates a web platform to interact with the webserver
"""
# =============================================================================
# Imports
# =============================================================================

import helper_dami_homework

import dash
from dash.dependencies import Input, Output, State

from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# =============================================================================
# Main
# =============================================================================

"""
VARIABLES TO FILL OUT

STUDENT_TRAINED_MODEL_FILENAME:
The filename specified should be located in the folder `HW3/`
"""
STUDENT_TRAINED_MODEL_FILENAME = "model_diabetes.pickle"

# Relative paths respect to current file
THIS_FILE_PATH = str(Path(__file__).parent.absolute())+"/../"
filename_to_load = THIS_FILE_PATH + STUDENT_TRAINED_MODEL_FILENAME

print('Loading pickle file with trained model from', filename_to_load)

# Variables to create the data structure from the web interface
dataset_colnames = None
sample = None   # DataFrame with the data that the user has input in the webpage

# Load trained model
loaded_model = None
with open(filename_to_load, "rb") as readFile:
    loaded_model = pickle.load(readFile)

### CREATE DASH APPLICATION

# Styling for HTML website
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create web server
app = dash.Dash("dami_homework", external_stylesheets=external_stylesheets)

# In the additional file `helper_dami_homework` is hidden all webpage' structure
app.layout = helper_dami_homework.app_html_layout

# =============================================================================
# Callbacks to setup the interaction between webpage and controls
# The next syntax is specific from the dash library, documentation can be found
# on https://dash.plotly.com/
# =============================================================================

# Sliders
@app.callback(
    Output(component_id='value-slider-Pregnancies', component_property='children'),
    [Input(component_id='slider-Pregnancies', component_property='value')]
)
def update_pregnancies(value):
    return str(value)

@app.callback(
    Output(component_id='value-slider-Glucose', component_property='children'),
    [Input(component_id='slider-Glucose', component_property='value')]
)
def update_glucose(value):
    return str(value)

@app.callback(
    Output(component_id='value-slider-BloodPressure', component_property='children'),
    [Input(component_id='slider-BloodPressure', component_property='value')]
)
def update_blood_pressure(value):
    return str(value)

@app.callback(
    Output(component_id='value-slider-SkinThickness', component_property='children'),
    [Input(component_id='slider-SkinThickness', component_property='value')]
)
def update_skin_thickness(value):
    return str(value)

@app.callback(
    Output(component_id='value-slider-Insulin', component_property='children'),
    [Input(component_id='slider-Insulin', component_property='value')]
)
def update_width_insulin(value):
    return str(value)

@app.callback(
    Output(component_id='value-slider-BMI', component_property='children'),
    [Input(component_id='slider-BMI', component_property='value')]
)
def update_asymm_BMI(value):
    return str(value)

@app.callback(
    Output(component_id='value-slider-DPF', component_property='children'),
    [Input(component_id='slider-DPF', component_property='value')]
)
def update_diabetes_pedigree_function(value):
    return str(value)

@app.callback(
    Output(component_id='value-slider-Age', component_property='children'),
    [Input(component_id='slider-Age', component_property='value')]
)
def update_age(value):
    return str(value)


# Visualization
@app.callback(
    Output(component_id='graph-histogram', component_property='figure'),
    [Input(component_id='dropdown-histogram', component_property='value'),
    Input(component_id='submit', component_property='n_clicks')]
)
def update_histogram(colname, n_clicks):
    return helper_dami_homework.update_histogram(colname, sample)

@app.callback(
    Output(component_id='graph-scatter', component_property='figure'),
    [Input(component_id='dropdown-scatter-1', component_property='value'),
    Input(component_id='dropdown-scatter-2', component_property='value'),
    Input(component_id='submit', component_property='n_clicks')]
)
def update_scatter(col1, col2, n_clicks):
    return helper_dami_homework.update_scatter(col1, col2, sample)


# Classification Button
@app.callback(    
    Output(component_id='classification-result', component_property='children'),
    [Input(component_id='submit', component_property='n_clicks')],
    [State('slider-Pregnancies', 'value'),
    State('slider-Glucose', 'value'),
    State('slider-BloodPressure', 'value'),
    State('slider-SkinThickness', 'value'),
    State('slider-Insulin', 'value'),
    State('slider-BMI', 'value'),
    State('slider-DPF', 'value'),
    State('slider-Age', 'value'),
    ]
)
def execute_classification(n_clicks, P, G, BP, ST, I, BMI, DPF, A):
    """
    Main method. Loads the trained model, applies the input data and returns a class
    """
    if(n_clicks == None): # When the application open
        return "Press below to execute the classification"
    else:
        # The sliders' values are already parsed to numeric values
        # Here we create a DataFrame with the input data
        data_from_user = [P, G, BP, ST, I, BMI, DPF, A]
        dataset_colnames = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        global sample
        sample = pd.DataFrame(data=[data_from_user], columns=dataset_colnames)

        # Execute the prediction using the loaded trained model.
        prediction = loaded_model.predict(sample)

        # Return final message
        prediction_labels = ["NO DIABETES", "WITH DIABETES"]
        return "The predicted class of the input data is: ["+ str(prediction[0]) +":" + prediction_labels[prediction[0]] + "]"


# Run the web server when this script is executed in Python
if __name__ == "__main__":
    app.run_server(debug=True)