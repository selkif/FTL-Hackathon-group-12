import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model_path = "../models/water_usage_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model not found at {model_path}. Please run train_model.py first.")

# Define all possible crop types (update this list based on your dataset)
CROP_TYPES = ["WHEAT", "MAIZE", "RICE", "SOYBEAN", "BARLEY", "POTATO", "BANANA", "MELON", "TOMATO", "ONION", "CABBAGE", "SUGARCANE", "COTTON", "MUSTARD", "BEAN", "CITRUS"] # Use uppercase to match the dataset

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None  # Initialize prediction as None
    if request.method == "POST":
        # Get user input and convert to uppercase to match the dataset
        min_temp = float(request.form["min_temp"])
        max_temp = float(request.form["max_temp"])
        soil_type = request.form["soil_type"].upper()  # Convert to uppercase
        crop_type = request.form["crop_type"].upper()  # Convert to uppercase
        weather_condition = request.form["weather_condition"].upper()  # Convert to uppercase
        region = request.form["region"].upper()  # Convert to uppercase

        # Create a DataFrame for the input data with all feature columns
        input_data = pd.DataFrame(columns=model.feature_names_in_)

        # Fill in the input data
        input_data.loc[0, 'MIN_TEMP'] = min_temp
        input_data.loc[0, 'MAX_TEMP'] = max_temp
        input_data.loc[0, 'Soil Type_' + soil_type] = 1
        input_data.loc[0, 'Crop Type_' + crop_type] = 1
        input_data.loc[0, 'Weather Condition_' + weather_condition] = 1
        input_data.loc[0, 'REGION_' + region] = 1

        # Fill missing columns with 0
        input_data = input_data.fillna(0)

        # Ensure the columns are in the correct order
        input_data = input_data[model.feature_names_in_]

        # Predict water requirement
        water_requirement = model.predict(input_data)[0]
        prediction = f"Recommended Water Usage: {water_requirement:.2f} mm"

    # Render the template with the prediction result
    return render_template("index.html", crop_types=CROP_TYPES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)