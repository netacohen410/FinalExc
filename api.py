# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:29:15 2023

@author: user
"""


#import numpy as np
from flask import Flask, request,render_template
import pickle
import pandas as pd
#import os

app = Flask(__name__)
rf_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_values = list(request.form.values())
    input_df = pd.DataFrame([input_values], columns=['City', 'type','Area','hasElevator ','hasParking ','condition ','hasBalcony ','hasMamad ','entrance_date'])
    prediction = rf_model.predict(input_df)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Price: {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)