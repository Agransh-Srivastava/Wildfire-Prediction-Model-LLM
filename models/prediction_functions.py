import pandas as pd
import numpy as np
import os
import joblib

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def load_model():
    model_path = os.path.join(MODEL_DIR, 'wildfire_prediction_model.joblib')
    feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.joblib')
    
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_columns_path)
    
    return model, feature_columns

def assign_risk_category(probability):
    if probability < 0.25:
        return 'Low'
    elif probability < 0.5:
        return 'Moderate'
    elif probability < 0.75:
        return 'High'
    else:
        return 'Extreme'

def get_grid_cells_for_location(location_name):
    """
    Get real grid cells based on location name and fire_frequency_by_grid.csv
    """
    grid_mapping_file = os.path.join(PROCESSED_DIR, 'fire_frequency_by_grid.csv')
    grid_data = pd.read_csv(grid_mapping_file)
    
    location_name = location_name.lower()
    
    if location_name == 'los angeles':
        region_grids = grid_data[
            (grid_data['lat'] > 33.5) & (grid_data['lat'] < 34.8) &
            (grid_data['lon'] > -119.0) & (grid_data['lon'] < -117.5)
        ]
    elif location_name == 'san francisco':
        region_grids = grid_data[
            (grid_data['lat'] > 37.0) & (grid_data['lat'] < 38.2) &
            (grid_data['lon'] > -123.0) & (grid_data['lon'] < -121.5)
        ]
    elif location_name == 'sacramento':
        region_grids = grid_data[
            (grid_data['lat'] > 38.3) & (grid_data['lat'] < 39.0) &
            (grid_data['lon'] > -122.5) & (grid_data['lon'] < -121.0)
        ]
    elif location_name == 'san diego':
        region_grids = grid_data[
            (grid_data['lat'] > 32.5) & (grid_data['lat'] < 33.3) &
            (grid_data['lon'] > -117.5) & (grid_data['lon'] < -116.5)
        ]
    elif location_name == 'california':
        region_grids = grid_data
    else:
        print(f"Warning: Unknown location '{location_name}', returning empty list.")
        return []
    
    return region_grids['grid_cell_id'].tolist()

def predict_fire_risk(grid_cell_id, future_date, weather_data=None, model=None, feature_columns=None):
    """
    Predict fire risk for a specific grid cell on a future date.
    """
    if model is None or feature_columns is None:
        model, feature_columns = load_model()
    
    date_obj = pd.to_datetime(future_date)
    
    pred_df = pd.DataFrame({
        'grid_cell_id': [grid_cell_id],
        'date': [date_obj],
        'year': [date_obj.year],
        'month': [date_obj.month]
    })
    
    pred_df['month_sin'] = np.sin(pred_df['month'] * (2 * np.pi / 12))
    pred_df['month_cos'] = np.cos(pred_df['month'] * (2 * np.pi / 12))
    
    if weather_data is not None:
        pred_df = pd.merge(pred_df, weather_data, on='date', how='left')
    
    # 🔥 Always restore month after merging (important!)
    pred_df['month'] = pred_df['date'].dt.month

    # Add month risk
    month_risk = {
        1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.5, 
        6: 0.7, 7: 0.9, 8: 1.0, 9: 0.8, 10: 0.4, 
        11: 0.2, 12: 0.1
    }
    pred_df['month_risk'] = pred_df['month'].map(month_risk)
    
    for col in feature_columns:
        if col not in pred_df.columns:
            pred_df[col] = 0  # Default 0 if missing
    
    X_pred = pred_df[feature_columns].fillna(0)
    
    probability = model.predict_proba(X_pred)[0, 1]
    
    risk_category = assign_risk_category(probability)
    
    return {
        'grid_cell_id': grid_cell_id,
        'date': future_date,
        'probability': probability,
        'fire_prediction': 1 if probability >= 0.5 else 0,
        'risk_category': risk_category,
        'month': date_obj.month
    }

def predict_risk_for_region(region_grid_cells, future_date, weather_data=None):
    """
    Predict fire risk for multiple grid cells representing a region.
    """
    model, feature_columns = load_model()
    results = []
    
    for grid_cell in region_grid_cells:
        prediction = predict_fire_risk(
            grid_cell_id=grid_cell,
            future_date=future_date,
            weather_data=weather_data,
            model=model,
            feature_columns=feature_columns
        )
        results.append(prediction)
    
    return pd.DataFrame(results)
