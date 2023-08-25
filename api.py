import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_crop_recommendation(N, K, P, pH, temperature):
    xgboost_model = joblib.load('RandomForest.pkl')
    data = {
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'ph': [pH],
    }
    input_data = pd.DataFrame(data)
    predictions = xgboost_model.predict(input_data)
    return predictions[0]  # Assuming the model returns a single prediction

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    N = data['nitrogen_level']
    K = data['potassium_level']
    P = data['phosphorus_level']
    pH = data['pH_level']
    temperature = data['temperature']

    crop_recommendation = get_crop_recommendation(N, K, P, pH, temperature)

    response = {
        "crop_recommendation": crop_recommendation
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
