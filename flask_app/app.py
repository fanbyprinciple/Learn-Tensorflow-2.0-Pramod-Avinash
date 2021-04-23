import pandas as pd 
import numpy as np 
import sklearn 
import joblib 
from flask import Flask, render_template, request

app  = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print(request.form.get('height'))
        try:
            height = float(request.form['height'])
            pred_args = np.array(height).reshape(-1,1)
            print("pred args: ", pred_args)
            model = open("heights_weights.pkl", 'rb')
            lr_model = joblib.load(model)
            model_prediction = lr_model.predict(pred_args)
            model_prediction = round(float(model_prediction),2)
            print(model_prediction)
        except ValueError:
            return "Please Enter valid values"
    return render_template('home.html', prediction=model_prediction, inputheight=height)
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')