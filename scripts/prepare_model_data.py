import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib

# Set up proper path handling for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading data from: {DATA_DIR}")

# Load all relevant datasets with correct paths
try:
    weather_fire_df = pd.read_csv(os.path.join(DATA_DIR, 'CA_Weather_Fire_Dataset_1984-2025.csv'))
    print(f"Loaded weather fire data: {weather_fire_df.shape}")
except Exception as e:
    print(f"Error loading weather fire data: {e}")
    weather_fire_df = pd.DataFrame()

try:
    fire_perimeters_df = pd.read_csv(os.path.join(DATA_DIR, 'California_Fire_Perimeters_(1950+).csv'))
    print(f"Loaded fire perimeters data: {fire_perimeters_df.shape}")
except Exception as e:
    print(f"Error loading fire perimeters data: {e}")
    fire_perimeters_df = pd.DataFrame()

try:
    incident_map_df = pd.read_csv(os.path.join(DATA_DIR, 'mapdataall.csv'))
    print(f"Loaded incident map data: {incident_map_df.shape}")
except Exception as e:
    print(f"Error loading incident map data: {e}")
    incident_map_df = pd.DataFrame()

# Convert date columns to datetime
print("Converting date columns...")

# Process weather_fire_df
if not weather_fire_df.empty:
    if 'DATE' in weather_fire_df.columns:
        weather_fire_df['DATE'] = pd.to_datetime(weather_fire_df['DATE'], errors='coerce')

# Process fire_perimeters_df
if not fire_perimeters_df.empty:
    date_columns = ['ALARM_DATE', 'CONT_DATE']
    for col in date_columns:
        if col in fire_perimeters_df.columns:
            fire_perimeters_df[col] = pd.to_datetime(fire_perimeters_df[col], errors='coerce')

# Process incident_map_df
if not incident_map_df.empty:
    date_columns = ['incident_date_last_update', 'incident_date_created', 'incident_date_extinguished']
    for col in date_columns:
        if col in incident_map_df.columns:
            incident_map_df[col] = pd.to_datetime(incident_map_df[col], errors='coerce')

# Create spatial grid for California
print("Creating spatial grid...")
# Define California's bounding box
CA_MIN_LAT, CA_MAX_LAT = 32.5, 42.0
CA_MIN_LON, CA_MAX_LON = -124.4, -114.1

# Create grid with 0.1° x 0.1° cells
grid_size = 0.1
lat_bins = np.arange(CA_MIN_LAT, CA_MAX_LAT + grid_size, grid_size)
lon_bins = np.arange(CA_MIN_LON, CA_MAX_LON + grid_size, grid_size)

# Function to assign grid cell ID
def assign_grid_cell(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    
    # Check if within California bounds
    if lat < CA_MIN_LAT or lat > CA_MAX_LAT or lon < CA_MIN_LON or lon > CA_MAX_LON:
        return np.nan
    
    lat_idx = int((lat - CA_MIN_LAT) / grid_size)
    lon_idx = int((lon - CA_MIN_LON) / grid_size)
    return lat_idx * len(lon_bins) + lon_idx

# Assign grid cells to incident_map_df
if not incident_map_df.empty and 'incident_latitude' in incident_map_df.columns and 'incident_longitude' in incident_map_df.columns:
    incident_map_df['grid_cell_id'] = incident_map_df.apply(
        lambda row: assign_grid_cell(row['incident_latitude'], row['incident_longitude']),
        axis=1
    )
    print(f"Assigned grid cells to {incident_map_df['grid_cell_id'].notna().sum()} incidents")

# Assign grid cells to fire_perimeters if it has lat/lon
# Note: This dataset might use centroids or need geometry processing
# This is a placeholder - adapt based on your actual data structure
if not fire_perimeters_df.empty:
    # Check if we have coordinates or need to calculate them
    if 'latitude' in fire_perimeters_df.columns and 'longitude' in fire_perimeters_df.columns:
        fire_perimeters_df['grid_cell_id'] = fire_perimeters_df.apply(
            lambda row: assign_grid_cell(row['latitude'], row['longitude']),
            axis=1
        )
    # If we don't have direct coordinates, but have geometry data (Shape__Area, Shape__Length)
    elif 'Shape__Area' in fire_perimeters_df.columns and 'Shape__Length' in fire_perimeters_df.columns:
        print("Note: Fire perimeters dataset has geometry but not direct coordinates. Additional processing needed.")
        # We could calculate centroids, but this would typically require more complex geometry processing

# Feature Engineering
print("Performing feature engineering...")

# 1. Prepare weather features
if not weather_fire_df.empty and 'DATE' in weather_fire_df.columns:
    # Create temporal features
    weather_fire_df['Month'] = weather_fire_df['DATE'].dt.month
    weather_fire_df['DayOfYear'] = weather_fire_df['DATE'].dt.dayofyear
    
    # Create cyclical encodings for temporal features
    weather_fire_df['month_sin'] = np.sin(weather_fire_df['Month'] * (2 * np.pi / 12))
    weather_fire_df['month_cos'] = np.cos(weather_fire_df['Month'] * (2 * np.pi / 12))
    weather_fire_df['dayofyear_sin'] = np.sin(weather_fire_df['DayOfYear'] * (2 * np.pi / 365))
    weather_fire_df['dayofyear_cos'] = np.cos(weather_fire_df['DayOfYear'] * (2 * np.pi / 365))
    
    # Sort by date for rolling calculations
    weather_fire_df = weather_fire_df.sort_values('DATE')
    
    # Calculate rolling averages for key weather metrics
    if 'PRECIPITATION' in weather_fire_df.columns:
        for window in [7, 14, 30]:
            weather_fire_df[f'rolling_{window}d_precip'] = weather_fire_df['PRECIPITATION'].rolling(window).sum()
            
    if 'MAX_TEMP' in weather_fire_df.columns and 'MIN_TEMP' in weather_fire_df.columns:
        for window in [7, 14, 30]:
            weather_fire_df[f'rolling_{window}d_max_temp'] = weather_fire_df['MAX_TEMP'].rolling(window).mean()
            weather_fire_df[f'rolling_{window}d_min_temp'] = weather_fire_df['MIN_TEMP'].rolling(window).mean()
            
    if 'AVG_WIND_SPEED' in weather_fire_df.columns:
        for window in [7, 14, 30]:
            weather_fire_df[f'rolling_{window}d_wind'] = weather_fire_df['AVG_WIND_SPEED'].rolling(window).mean()
    
    # Create drought indicators
    if 'PRECIPITATION' in weather_fire_df.columns:
        weather_fire_df['dry_day'] = (weather_fire_df['PRECIPITATION'] < 0.01).astype(int)
        weather_fire_df['consecutive_dry_days'] = (weather_fire_df['dry_day']
                                                  .groupby((weather_fire_df['dry_day'] != weather_fire_df['dry_day'].shift()).cumsum())
                                                  .cumsum() * weather_fire_df['dry_day'])

    # Create fire danger indices
    if all(col in weather_fire_df.columns for col in ['MAX_TEMP', 'AVG_WIND_SPEED', 'PRECIPITATION']):
        # Simple fire danger index: high temp + high wind + low precip
        weather_fire_df['fire_danger_index'] = (
            weather_fire_df['MAX_TEMP'] / 100 +  # Normalize around 1
            weather_fire_df['AVG_WIND_SPEED'] / 10 -  # Normalize around 1
            weather_fire_df['PRECIPITATION'] * 5  # Negative contribution, amplified
        )
        
        # Normalize the index to 0-100 scale
        scaler = MinMaxScaler(feature_range=(0, 100))
        weather_fire_df['fire_danger_index'] = scaler.fit_transform(
            weather_fire_df[['fire_danger_index']]
        )

# 2. Process and merge fire occurrence data
print("Processing fire occurrence data...")

# Create a time series grid dataset for modeling
# This will have one row per grid cell per time period

# Determine modeling time granularity (e.g., monthly)
time_granularity = 'ME'  # 'D' for daily, 'ME' for month end (replacing deprecated 'M')

# Create a complete date range for our dataset
if not weather_fire_df.empty and 'DATE' in weather_fire_df.columns:
    min_date = weather_fire_df['DATE'].min()
    max_date = weather_fire_df['DATE'].max()
    
    if pd.notna(min_date) and pd.notna(max_date):
        # Use monthly date range for prediction model
        date_range = pd.date_range(start=min_date, end=max_date, freq=time_granularity)
        
        # Create grid-date combinations
        grid_cells = np.arange(len(lat_bins) * len(lon_bins))
        
        # Create all combinations of grid cells and dates
        grid_dates = []
        for cell_id in grid_cells:
            for date in date_range:
                grid_dates.append({
                    'grid_cell_id': cell_id,
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'fire_occurred': 0  # Default: no fire
                })
        
        grid_date_df = pd.DataFrame(grid_dates)
        
        # Mark which grid cells had fires at which times
        if not incident_map_df.empty and 'grid_cell_id' in incident_map_df.columns:
            for idx, row in incident_map_df.iterrows():
                if pd.isna(row['grid_cell_id']) or pd.isna(row['incident_date_created']):
                    continue
                
                # Find the period in grid_date_df that matches this fire
                date = row['incident_date_created']
                period_start = date.replace(day=1) if time_granularity == 'M' else date
                
                # Mark this grid cell as having a fire in this period
                match_idx = grid_date_df[
                    (grid_date_df['grid_cell_id'] == row['grid_cell_id']) &
                    (grid_date_df['date'] == period_start)
                ].index
                
                if len(match_idx) > 0:
                    grid_date_df.loc[match_idx, 'fire_occurred'] = 1
        
        # Similarly process fire_perimeters_df if it has the necessary date and location data
        
        # Merge weather data with the grid-date DataFrame
        # For each grid cell, find the appropriate weather based on location and date
        # This is a simplification - in a real implementation you'd need to:
        # 1. Have weather data assigned to grid cells
        # 2. Perform a more sophisticated join
        
        # For now, we'll use global weather as a placeholder
        # In reality, you'd need gridded weather data or weather station data mapped to grid cells
        if not weather_fire_df.empty and 'DATE' in weather_fire_df.columns:
            weather_monthly = weather_fire_df.copy()
            if time_granularity == 'ME':
                # Convert DATE to month-end dates for grouping
                weather_monthly['date'] = weather_monthly['DATE'].dt.to_period('M').dt.to_timestamp(how='end')
                
                # Separate numeric and non-numeric columns to handle them differently
                all_features = [col for col in weather_monthly.columns if col not in ['DATE', 'date']]
                
                # Check each column's data type
                numeric_features = []
                categorical_features = []
                
                for col in all_features:
                    # Try to convert to numeric to test if it's actually numeric
                    try:
                        pd.to_numeric(weather_monthly[col])
                        numeric_features.append(col)
                    except (ValueError, TypeError):
                        categorical_features.append(col)
                
                print(f"Numeric features: {numeric_features}")
                print(f"Categorical features: {categorical_features}")
                
                # Initialize an empty dataframe with just the dates
                unique_dates = weather_monthly['date'].unique()
                weather_agg = pd.DataFrame({'date': unique_dates})
                
                # Process numeric features with mean
                if numeric_features:
                    for col in numeric_features:
                        # Get mean for each date and merge back
                        temp = weather_monthly.groupby('date')[col].mean().reset_index()
                        weather_agg = pd.merge(weather_agg, temp, on='date', how='left')
                
                # Process categorical features with first value
                if categorical_features:
                    for col in categorical_features:
                        # Get first value for each date and merge back
                        temp = weather_monthly.groupby('date')[col].first().reset_index()
                        weather_agg = pd.merge(weather_agg, temp, on='date', how='left')
                
                # Replace weather_monthly with the aggregated dataframe
                weather_monthly = weather_agg
                
                print(f"Weather monthly dataframe has {len(weather_monthly)} rows and these columns: {weather_monthly.columns.tolist()}")
                print(f"First few dates: {weather_monthly['date'].head().tolist()}")
            
            # Merge weather with grid-date DataFrame
            grid_date_df = pd.merge(
                grid_date_df,
                weather_monthly,
                on='date',
                how='left'
            )
        
        print(f"Created grid-date dataset with {len(grid_date_df)} rows")
        
        # Save the prepared dataset
        output_file = os.path.join(PROCESSED_DIR, 'wildfire_prediction_data.csv')
        grid_date_df.to_csv(output_file, index=False)
        print(f"Saved prepared dataset to {output_file}")
        
        # Create train-test split
        print("Creating train-test split...")
        
        # Use data up to 2020 for training
        train_data = grid_date_df[grid_date_df['year'] <= 2020]
        # Use 2021-2023 for validation
        val_data = grid_date_df[(grid_date_df['year'] >= 2021) & (grid_date_df['year'] <= 2023)]
        # Reserve 2024-2025 for testing
        test_data = grid_date_df[grid_date_df['year'] >= 2024]
        
        print(f"Training data: {len(train_data)} rows")
        print(f"Validation data: {len(val_data)} rows")
        print(f"Test data: {len(test_data)} rows")
        
        # Save the splits
        train_data.to_csv(os.path.join(PROCESSED_DIR, 'train_data.csv'), index=False)
        val_data.to_csv(os.path.join(PROCESSED_DIR, 'val_data.csv'), index=False)
        test_data.to_csv(os.path.join(PROCESSED_DIR, 'test_data.csv'), index=False)

# Optional: Create a spatial visualization of fire frequency
if not incident_map_df.empty and 'grid_cell_id' in incident_map_df.columns:
    fire_counts = incident_map_df['grid_cell_id'].value_counts().reset_index()
    fire_counts.columns = ['grid_cell_id', 'fire_count']
    
    # Function to convert grid_cell_id back to lat/lon
    def grid_cell_to_latlon(cell_id):
        lon_count = len(lon_bins) - 1
        lat_idx = cell_id // lon_count
        lon_idx = cell_id % lon_count
        
        lat = CA_MIN_LAT + lat_idx * grid_size
        lon = CA_MIN_LON + lon_idx * grid_size
        return lat, lon
    
    # Add lat/lon back to fire_counts
    fire_counts['lat'] = fire_counts['grid_cell_id'].apply(lambda x: grid_cell_to_latlon(x)[0])
    fire_counts['lon'] = fire_counts['grid_cell_id'].apply(lambda x: grid_cell_to_latlon(x)[1])
    
    # Save the spatial fire frequency data
    fire_counts.to_csv(os.path.join(PROCESSED_DIR, 'fire_frequency_by_grid.csv'), index=False)
    
    print(f"Created spatial fire frequency dataset with {len(fire_counts)} cells")

print("Data preparation complete!")