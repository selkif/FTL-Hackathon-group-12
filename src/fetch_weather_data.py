import os

import pandas as pd
import requests


def fetch_weather_data(lat, lon):
    """
    Fetch weather data from Open-Meteo API for a given latitude and longitude.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=temperature_2m,relativehumidity_2m,precipitation"
    response = requests.get(url)
    data = response.json()

    
    weather_data = {
        'LATITUDE': lat,
        'LONGITUDE': lon,
        'TEMPERATURE': data['current_weather']['temperature'],
        'RAINFALL': data['hourly']['precipitation'][0],  # Rainfall in the last hour (mm)
        'HUMIDITY': data['hourly']['relativehumidity_2m'][0]  # Humidity in percentage
    }
    return weather_data


lat = 9.1450
lon = 40.4897
weather_data = fetch_weather_data(lat, lon)


weather_df = pd.DataFrame([weather_data])

os.makedirs("../data/external", exist_ok=True)


weather_df.to_csv("../data/external/weather_data.csv", index=False)

print("Weather data saved to data/external/weather_data.csv")