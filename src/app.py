import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model, scaler, and feature names
model_path = "../models/water_usage_model.pkl"
scaler_path = "../models/scaler.pkl"
feature_names_path = "../models/feature_names.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the model is placed in the 'models/' folder.")

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please ensure the scaler is placed in the 'models/' folder.")

if os.path.exists(feature_names_path):
    FEATURE_NAMES = joblib.load(feature_names_path)
else:
    raise FileNotFoundError(f"Feature names not found at {feature_names_path}. Please ensure the feature names are placed in the 'models/' folder.")


CROP_TYPES = ["WHEAT", "MAIZE", "RICE", "SOYBEAN", "BARLEY", "POTATO", "BANANA", "MELON", "TOMATO", "ONION", "CABBAGE", "SUGARCANE", "COTTON", "MUSTARD", "BEAN", "CITRUS"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Get user inputs from the form
        min_temp = float(request.form["min_temp"])
        max_temp = float(request.form["max_temp"])
        soil_type = request.form["soil_type"].upper()
        crop_type = request.form["crop_type"].upper()
        weather_condition = request.form["weather_condition"].upper()
        region = request.form["region"].upper()

        # Create a DataFrame for the input data
        input_data = pd.DataFrame(columns=FEATURE_NAMES)

        # Fill in the input data
        input_data.loc[0, 'MIN_TEMP'] = min_temp
        input_data.loc[0, 'MAX_TEMP'] = max_temp
        input_data.loc[0, f'Crop Type_{crop_type}'] = 1
        input_data.loc[0, f'Soil Type_{soil_type}'] = 1
        input_data.loc[0, f'Weather Condition_{weather_condition}'] = 1
        input_data.loc[0, f'REGION_{region}'] = 1

        
        input_data = input_data.fillna(0)

        # Ensure the columns match the model's expected features
        input_data = input_data[FEATURE_NAMES]

        
        input_data_scaled = scaler.transform(input_data)

        
        water_requirement = model.predict(input_data_scaled)[0]
        prediction = f"Recommended Water Usage: {water_requirement:.2f} mm"

    
    return render_template("index.html", crop_types=CROP_TYPES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)