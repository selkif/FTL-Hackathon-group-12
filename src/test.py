import pandas as pd

# Load the final dataset
data = pd.read_csv("../data/processed/final_dataset.csv")

# Check the data types of each column
print(data.dtypes)

# Display the first few rows of the dataset
print(data.head())