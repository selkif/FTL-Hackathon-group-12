import os

import pandas as pd


def preprocess_data():
    """
    Preprocess DATASET - Sheet1.csv and save the final dataset for training.
    """
    # Create the processed data directory if it doesn't exist
    os.makedirs("../data/processed", exist_ok=True)

    # Load the dataset
    dataset_path = "../data/raw/DATASET - Sheet1.csv"
    data = pd.read_csv(dataset_path)

    # Rename columns for consistency
    data.rename(columns={
        'CROP TYPE': 'Crop Type',
        'SOIL TYPE': 'Soil Type',
        'TEMPERATURE': 'Temperature',
        'WEATHER CONDITION': 'Weather Condition',
        'WATER REQUIREMENT': 'Water Requirement'
    }, inplace=True)

    # Split the Temperature column into MIN_TEMP and MAX_TEMP
    data[['MIN_TEMP', 'MAX_TEMP']] = data['Temperature'].str.split('-', expand=True).astype(float)
    data.drop(columns=['Temperature'], inplace=True)

    # Encode categorical columns
    categorical_columns = ['Crop Type', 'Soil Type', 'Weather Condition', 'REGION']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Save the final dataset
    final_data_path = "../data/processed/final_dataset.csv"
    data.to_csv(final_data_path, index=False)
    print(f"Final dataset saved to {final_data_path}")


if __name__ == "__main__":
    preprocess_data()