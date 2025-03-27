from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app, origins="*")

# OpenWeather API details
API_KEY = os.getenv("API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5"

# Load saved models
with open("./models/rain_model.pkl", "rb") as f:
    rain_model = pickle.load(f)
with open("./models/temp_model.pkl", "rb") as f:
    temp_model = pickle.load(f)
with open("./models/hum_model.pkl", "rb") as f:
    hum_model = pickle.load(f)
with open("./models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Fetch live weather data
def get_current_weather(city):
    url = f"{BASE_URL}/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None  # Return None if API call fails

    data = response.json()
    return {
        "MinTemp": round(data["main"]["temp_min"]),
        "MaxTemp": round(data["main"]["temp_max"]),
        "WindGustDir": data["wind"]["deg"],
        "Humidity": data["main"]["humidity"],
        "Pressure": data["main"]["pressure"],
        "Temp": round(data["main"]["temp"]),
        "WindGustSpeed": data["wind"]["speed"]
    }

# Predict future values
def predict_future(model, current_value, feature_name):
    df = pd.DataFrame({feature_name: [current_value]})  # Ensure feature names match training
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(df)[0]  # Predict with correct format
        predictions.append(next_value)
        df = pd.DataFrame({feature_name: [next_value]})  # Update input format for next step
    return predictions[1:]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or "city" not in data:
        return jsonify({"error": "City name is required"}), 400

    city = data["city"]
    current_weather = get_current_weather(city)
    
    if current_weather is None:
        return jsonify({"error": "Invalid city name or API issue"}), 400

    # Prepare current data for prediction
    wind_deg = current_weather['WindGustDir'] % 360
    compass_points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    compass_direction = compass_points[int(wind_deg / 22.5) % 16]
    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

    current_data = {
        "MinTemp": current_weather['MinTemp'],
        "MaxTemp": current_weather['MaxTemp'],
        "WindGustDir": compass_direction_encoded,
        "WindGustSpeed": current_weather['WindGustSpeed'],
        "Humidity": current_weather['Humidity'],
        "Pressure": current_weather['Pressure'],
        "Temp": current_weather['Temp']
    }

    current_df = pd.DataFrame([current_data])[["MinTemp", "MaxTemp", "WindGustDir", "Humidity", "Pressure", "Temp", "WindGustSpeed"]]

    # Make predictions
    rain_prediction = rain_model.predict(current_df)[0]
    future_temp = predict_future(temp_model, current_weather['Temp'], "Temp")
    future_hum = predict_future(hum_model, current_weather['Humidity'], "Humidity")


    # Generate future timestamps
    timezone = pytz.timezone("Asia/Kolkata")
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

    return jsonify({
        "city": city,
        "current_temperature": current_weather['Temp'],
        "humidity": current_weather['Humidity'],
        "rain_prediction": "Yes" if rain_prediction else "No",
        "future_temperatures": dict(zip(future_times, map(round, future_temp))),
        "future_humidities": dict(zip(future_times, map(round, future_hum)))
    })

if __name__ == '__main__':
    app.run(debug=True)
