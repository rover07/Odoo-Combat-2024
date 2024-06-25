from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model which includes the preprocessor
with open('models/RidgeModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract locations for the dropdown in the web form
data = pd.read_csv('notebook/data/Cleaned_data.csv')
__locations = sorted(data['location'].unique().tolist())

def get_estimated_price(input_json):
    # Create a dataframe for the input
    input_df = pd.DataFrame([input_json])

    # Predict the price using the loaded model
    result = round(model.predict(input_df)[0], 2)
    return result

@app.route('/')
def index():
    return render_template('index.html', locations=__locations)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_json = {
            "location": request.form['sLocation'],
            "total_sqft": float(request.form['Squareft']),
            "bhk": int(request.form['uiBHK']),
            "bath": int(request.form['uiBathrooms'])
        }
        result = get_estimated_price(input_json)

        if result > 100:
            result = round(result / 100, 2)
            result = str(result) + ' Crore'
        else:
            result = str(result) + ' Lakhs'

    return render_template('prediction.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
