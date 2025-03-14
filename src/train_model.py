import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_model():
    """
    Train a machine learning model and save it to the models folder.
    """
    # Create the models directory if it doesn't exist
    os.makedirs("../models", exist_ok=True)

    # Load the preprocessed dataset
    data_path = "../data/processed/final_dataset.csv"
    data = pd.read_csv(data_path)

    # Check for missing values in the target column
    print("Missing values in target column (Water Requirement):", data['Water Requirement'].isnull().sum())

    # Split into features (X) and target (y)
    X = data.drop(columns=['Water Requirement'])  # Features
    y = data['Water Requirement']  # Target

    # Check if the dataset is empty
    if len(data) == 0:
        raise ValueError("The dataset is empty after preprocessing. Check the preprocessing steps.")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42, n_estimators=100)  # You can tune hyperparameters
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Save the model
    model_path = "../models/water_usage_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()