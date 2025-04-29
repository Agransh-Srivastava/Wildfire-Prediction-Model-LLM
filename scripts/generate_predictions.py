import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import sys

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'data', 'predictions')

# Add the MODEL_DIR to Python path
sys.path.append(MODEL_DIR)

# Import prediction functions
from prediction_functions import predict_fire_risk, get_grid_cells_for_location, predict_risk_for_region

# Create predictions directory if it doesn't exist
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Load model and feature columns
model = joblib.load(os.path.join(MODEL_DIR, 'wildfire_prediction_model.joblib'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.joblib'))

# Load grid mapping
grid_data = pd.read_csv(os.path.join(PROCESSED_DIR, 'fire_frequency_by_grid.csv'))

# Generate synthetic weather forecast
def generate_forecast_weather(start_date, months=12):
    forecast_dates = pd.date_range(start=start_date, periods=months, freq='MS')
    
    month_temp_pattern = {1:50, 2:55, 3:60, 4:65, 5:70, 6:80, 7:90, 8:90, 9:85, 10:75, 11:60, 12:50}
    month_precip_pattern = {1:4.0, 2:4.0, 3:3.0, 4:1.5, 5:0.5, 6:0.1, 7:0.0, 8:0.0, 9:0.2, 10:1.0, 11:2.0, 12:4.0}
    month_wind_pattern = {1:6, 2:7, 3:8, 4:7, 5:6, 6:5, 7:4, 8:4, 9:5, 10:6, 11:6, 12:6}
    
    weather_data = pd.DataFrame({'date': forecast_dates})
    weather_data['month'] = weather_data['date'].dt.month
    
    weather_data['MAX_TEMP'] = weather_data['month'].map(month_temp_pattern) + np.random.normal(0, 3, len(weather_data))
    weather_data['MIN_TEMP'] = weather_data['MAX_TEMP'] - 20 + np.random.normal(0, 2, len(weather_data))
    weather_data['PRECIPITATION'] = weather_data['month'].map(month_precip_pattern) * np.random.uniform(0.7, 1.3, len(weather_data))
    weather_data['AVG_WIND_SPEED'] = weather_data['month'].map(month_wind_pattern) + np.random.normal(0, 1, len(weather_data))
    
    weather_data['TEMP_RANGE'] = weather_data['MAX_TEMP'] - weather_data['MIN_TEMP']
    weather_data['WIND_TEMP_RATIO'] = weather_data['AVG_WIND_SPEED'] / (weather_data['MAX_TEMP'] + 0.1)
    
    return weather_data

# Predict for all California regions
def predict_california_regions(forecast_date, weather_data):
    major_regions = {
        'Northern California': grid_data[(grid_data['lat'] > 38.0)]['grid_cell_id'].tolist(),
        'Central California': grid_data[(grid_data['lat'] > 36.0) & (grid_data['lat'] <= 38.0)]['grid_cell_id'].tolist(),
        'Southern California': grid_data[(grid_data['lat'] <= 36.0)]['grid_cell_id'].tolist()
    }
    
    all_predictions = []
    
    for region_name, grid_cells in major_regions.items():
        region_predictions = predict_risk_for_region(grid_cells, forecast_date, weather_data)
        if not region_predictions.empty:
            region_summary = {
                'region': region_name,
                'date': forecast_date,
                'avg_probability': region_predictions['probability'].mean(),
                'max_probability': region_predictions['probability'].max(),
                'high_risk_percentage': (region_predictions['risk_category'].isin(['High', 'Extreme'])).mean() * 100
            }
            all_predictions.append(region_summary)
    
    return pd.DataFrame(all_predictions)

# Main
print("Generating 12-month forecast...")

forecast_start = datetime.now()
forecast_months = 12
forecast_weather = generate_forecast_weather(forecast_start, forecast_months)

monthly_predictions = []

for i in range(forecast_months):
    forecast_date = (forecast_start + timedelta(days=30*i)).strftime('%Y-%m-%d')
    
    # Correct month handling
    target_month = (forecast_start.month + i - 1) % 12 + 1
    
    month_weather = forecast_weather[forecast_weather['date'].dt.month == target_month].copy()
    
    # 🔥 Fix: make sure 'month' column exists
    if month_weather.empty:
        month_weather = pd.DataFrame({
            'date': [pd.to_datetime(forecast_date)],
            'month': [target_month],
            'MAX_TEMP': [70],
            'MIN_TEMP': [50],
            'PRECIPITATION': [1.0],
            'AVG_WIND_SPEED': [5],
            'TEMP_RANGE': [20],
            'WIND_TEMP_RATIO': [0.07]
        })
    else:
        month_weather['month'] = month_weather['date'].dt.month

    region_predictions = predict_california_regions(forecast_date, month_weather)
    monthly_predictions.append(region_predictions)

if monthly_predictions:
    all_predictions = pd.concat(monthly_predictions)
    forecast_file = os.path.join(PREDICTIONS_DIR, 'california_12month_forecast.csv')
    all_predictions.to_csv(forecast_file, index=False)
    print(f"Saved forecast to {forecast_file}")

    # Plot forecast
    plt.figure(figsize=(12, 6))
    for region in all_predictions['region'].unique():
        region_data = all_predictions[all_predictions['region'] == region]
        plt.plot(region_data['date'], region_data['avg_probability'], marker='o', label=region)
    plt.title('California Wildfire Risk Forecast (12 Months)')
    plt.xlabel('Month')
    plt.ylabel('Average Risk Probability')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    chart_file = os.path.join(PREDICTIONS_DIR, 'risk_forecast_chart.png')
    plt.savefig(chart_file)
    print(f"Saved risk forecast chart to {chart_file}")

print("✅ Step 3 (generate_predictions.py) update complete!")
