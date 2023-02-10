#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : luis-eduardo@dsv.su.se
# Created Date: 2020/06/30
# =============================================================================
"""
Creation of HTML webpage with Dash and visualization with Plotly.
This file is called from the `dami_analytics.py`, and its main goal
is to make the main code more readable.
"""
# =============================================================================
# Imports
# =============================================================================

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px

from pathlib import Path
import pandas as pd


# =============================================================================
# Functions
# =============================================================================

def update_histogram(colname = None, sample=None):
    """
    Draws a histogram plot from the original dataset and puts the value
    from the `sample` that the user has input in the website.
    """
    fig = px.histogram(data,
                    x=colname,
                    color="Outcome",
                    labels={k:v for k,v in zip(colnames,column_labels)},
                    template="ggplot2")
    fig.update_layout(
        legend = dict(title="Class",
                orientation="h",
                y=1, yanchor="bottom",
                x=0.5, xanchor="center"
                )
    )
    # Show a black line with the current value of the sample
    if (sample is not None):
        fig.add_shape(type="line", line_color="black",
                    line_width = 3, 
                    xref='x', yref='paper',
                    x0 = float(sample[colname]), x1 = float(sample[colname]),
                    y0 = 0, y1 = 1)
    return fig

def update_scatter(col1=None, col2=None, sample=None):
    """
    Draws a scatter plot from the original dataset and puts the value
    from the `sample` that the user has input in the website.
    """
    fig = px.scatter(data,
                    x=col1, 
                    y=col2, 
                    color="Outcome",
                    labels={k:v for k,v in zip(colnames,column_labels)},
                    template="simple_white")

    fig.update_layout(
        legend = dict(
                    title="Class",
                )
    )
     
    if (sample is not None):
        fig.add_annotation( # add a text callout with arrow
            text="SAMPLE!", x=float(sample[col1]), y=float(sample[col2]),
            arrowhead=3, showarrow=True, startarrowsize=3
        )
    return fig

# =============================================================================
# Main
# =============================================================================


#############
"""
Load and simple processing of the original dataset for visualization
purposes in the web application.
"""

# Relative paths respect to current file
# DO NOT MODIFY: Relative path prefix to be able to find the dataset
THIS_FILE_PATH = str(Path(__file__).parent.absolute())+"/"
FOLDER_PATH = THIS_FILE_PATH + "./"

# Load original dataset file
dataset_filename = FOLDER_PATH + "diabetes.csv"
data = pd.read_csv(dataset_filename)

# Structure to map df column names to meaningful labels
colnames = data.columns
colnames = colnames.drop('Outcome').values
column_labels = colnames # No need to map colnames to other labels

# Initialization of plots when the website is loaded the first time
fig_histogram = update_histogram("Pregnancies")
fig_scatter = update_scatter("Pregnancies","Glucose")

#############
"""
Structure of the HTML webpage using Dash library
"""
app_html_layout = html.Div([

    html.Center(html.H1("HW3 - DAMI - Diabetes Classification")),

    html.Div("This app predicts whether or not a patient has diabetes, based on diagnostic measurements. "),

    html.Div(['More information about dataset:', 
        html.A(' https://www.kaggle.com/uciml/pima-indians-diabetes-database/data')
    ]),

    html.H3('Classification with Trained Model'),

    # Create the table to put input values
    html.Table([ html.Tbody([
        # Pregnancies
        html.Tr([
            html.Td( html.B('Pregnancies (#):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-Pregnancies',
                    min=0,
                    max=20,
                    step=1,
                    value=1,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-Pregnancies',children=''), style={'width':'10%'} ),
            ]),
        # Glucose
        html.Tr([
            html.Td( html.B('Glucose (GTT):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-Glucose',
                    min=30,
                    max=200,
                    step=2,
                    value=120,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-Glucose',children=''), style={'width':'20%'} ),
            ]),
        # Blood Pressure
        html.Tr([
            html.Td( html.B('Diastolic Blood Pressure (mm Hg):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-BloodPressure',
                    min=40,
                    max=150,
                    step=1,
                    value=72,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-BloodPressure',children=''), style={'width':'20%'} ),
            ]),
        # Skin Thickness
        html.Tr([
            html.Td( html.B('Triceps Skin Thickness (mm):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-SkinThickness',
                    min=0,
                    max=100,
                    step=1,
                    value=32,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-SkinThickness',children=''), style={'width':'20%'} ),
            ]),
        # Insulin
        html.Tr([
            html.Td( html.B('Serum Insulin (uU/mL):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-Insulin',
                    min=0,
                    max=300,
                    step=2,
                    value=30,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-Insulin',children=''), style={'width':'20%'} ),
            ]),
        # BMI
        html.Tr([
            html.Td( html.B('BMI (kg/m<sup>2</sup>):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-BMI',
                    min=10,
                    max=60,
                    step=0.5,
                    value=32,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-BMI',children=''), style={'width':'20%'} ),
            ]),
        # Diabetes Pedigree Function
        html.Tr([
            html.Td( html.B('Diabetes Pedigree Function:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-DPF',
                    min=0,
                    max=3,
                    step=0.05,
                    value=0.37,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-DPF',children=''), style={'width':'20%'} ),
            ]),
        # Age
        html.Tr([
            html.Td( html.B('Age (years):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-Age',
                    min=20,
                    max=90,
                    step=1,
                    value=29,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-Age',children=''), style={'width':'20%'} ),
            ]),
        ]), 
    ], style={'width':'100%', 'padding':'0', 'margin':'0'}),

    html.Center( 
        html.Div([
            html.Br(),
            html.H4(html.B('Classification result', id='classification-result', style={'color':'#983e0f'})),
            html.Button('Execute Classification', id='submit', style={'margin':'0 auto', 'width':'30%'}),
        ])
    ),

    html.Br(),

    html.Center(html.B('Possible classes: [0:NO DIABETES], [1:WITH DIABETES]', style={'color':'blue'})),

    html.Hr(),

    html.H3('Dataset Visualization'),

    html.Div('The next plots show some characteristics of the original dataset. Note that the values from the SAMPLE that was input above will be highlighted in the plot according to the selected variables.'),

    # Layout for plots
    html.Table([
        html.Tbody([
            # Create the cell for the first plot
            html.Tr([
                html.Td([
                    
                    html.H5('Histogram per class of a variable'),

                    html.Label("Choose a variable:"),

                    dcc.Dropdown(id='dropdown-histogram',
                        options=[{"label":l, 'value':v} for l,v in zip(column_labels,colnames)],
                        value='Pregnancies'
                    ),

                    dcc.Graph(
                        id='graph-histogram',
                        figure = fig_histogram
                    ),
                ], style={'width':'40%'} ),

                html.Td([
                    
                    html.H5('Scatter plot of two variables'),

                    html.Label("Choose two variables to plot:"),

                    dcc.Dropdown(id='dropdown-scatter-1',
                        options=[{"label":l, 'value':v} for l,v in zip(column_labels,colnames)],
                        value='Glucose'
                    ),

                    dcc.Dropdown(id='dropdown-scatter-2',
                        options=[{"label":l, 'value':v} for l,v in zip(column_labels,colnames)],
                        value='BloodPressure'
                    ),

                    dcc.Graph(
                        id='graph-scatter',
                        figure = fig_scatter
                    ),
                ], style={'width':'60%'} )
            ])
        ])
    ], style={'width': '100%'}),
    
], style={'columnCount': 1, 'marginBottom': 50, 'marginTop': 50, 'marginLeft':120, 'marginRight':120}
)
