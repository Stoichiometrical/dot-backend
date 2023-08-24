import os
import openai
import joblib
import pandas as pd
from langchain import PromptTemplate
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set your OpenAI API key here
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = "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"

def get_crop_recommendation(N, K, P, pH, temperature):
    xgboost_model = joblib.load('RandomForest.pkl')
    # Prepare the input data as a DataFrame for the XGBoost model
    data = {
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'ph': [pH],
    }
    input_data = pd.DataFrame(data)

    # Make predictions using the XGBoost model
    predictions = xgboost_model.predict(input_data)

    return predictions[0]  # Assuming the model returns a single prediction

def generate_explanation(prompt):
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=150)
    return response["choices"][0]["text"]

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    N = data['nitrogen_level']
    K = data['potassium_level']
    P = data['phosphorus_level']
    pH = data['pH_level']
    temperature = data['temperature']

    crop_recommendation = get_crop_recommendation(N, K, P, pH, temperature)

    prompt_template = PromptTemplate.from_template(
        "A machine learning model predicted with 98% accuracy that based on the following input values:\n\n- Nitrogen level: {N}\n- Potassium level: {K}\n- Phosphorus level: {P}\n- pH level: {pH}\n- Temperature: {temperature}\n\nThe best crop to plant for these conditions is: {plant_recommendation}.\n\nPlease provide a detailed explanation, approximately 100 words, supporting the recommendation and highlighting how each of the input factors contributes to the suitability of the chosen crop for optimal growth and yield under the given conditions."
    )
    formatted_prompt = prompt_template.format(N=N, K=K, P=P, pH=pH, temperature=temperature, plant_recommendation=crop_recommendation)
    explanation = generate_explanation(formatted_prompt)

    response = {
        "crop_recommendation": crop_recommendation,
        "explanation": explanation
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
