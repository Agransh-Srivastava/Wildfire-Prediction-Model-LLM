import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(MODEL_DIR, 'results')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading preprocessed datasets...")

# Load the datasets
try:
    train_data = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(PROCESSED_DIR, 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_data.csv'))
    
    print(f"Loaded training data: {train_data.shape}")
    print(f"Loaded validation data: {val_data.shape}")
    print(f"Loaded test data: {test_data.shape}")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Load grid latitude mapping
fire_frequency_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'fire_frequency_by_grid.csv'))
grid_latitudes = fire_frequency_df.set_index('grid_cell_id')['lat'].to_dict()

# Add latitude to train/val/test
for df in [train_data, val_data, test_data]:
    df['latitude'] = df['grid_cell_id'].map(grid_latitudes)

# Check for fire occurrences
print(f"Number of fire occurrences in training data: {train_data['fire_occurred'].sum()}")
print(f"Number of fire occurrences in validation data: {val_data['fire_occurred'].sum()}")
print(f"Number of fire occurrences in test data: {test_data['fire_occurred'].sum()}")

# Create synthetic fire data
print("\nCreating synthetic fire data for training...")

# Define month risk
month_risk = {
    1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.5, 
    6: 0.7, 7: 0.9, 8: 1.0, 9: 0.8, 10: 0.4, 
    11: 0.2, 12: 0.1
}

# Define latitude risk
def lat_risk(lat):
    if pd.isna(lat):
        return 0.5  # Default medium risk
    if lat > 38:
        return 0.3  # Northern California
    elif lat > 36:
        return 0.6  # Central California
    else:
        return 1.0  # Southern California

# Calculate combined risk
train_data['month_risk'] = train_data['month'].map(month_risk)
train_data['lat_risk'] = train_data['latitude'].apply(lat_risk)
train_data['combined_risk'] = train_data['month_risk'] * train_data['lat_risk']

# How many synthetic fires to create
num_synthetic_fires = int(train_data.shape[0] * 0.005)
num_synthetic_fires = max(num_synthetic_fires, 1000)
print(f"Creating {num_synthetic_fires} synthetic fires")

# Probability sampling
prob = train_data['combined_risk'] / train_data['combined_risk'].sum()
fire_indices = np.random.choice(
    train_data.index,
    size=num_synthetic_fires,
    replace=False,
    p=prob
)

# Set these as fires
train_data.loc[fire_indices, 'fire_occurred'] = 1

# Same for validation data (smaller amount)
val_data['month_risk'] = val_data['month'].map(month_risk)
val_data['lat_risk'] = val_data['latitude'].apply(lat_risk)
val_data['combined_risk'] = val_data['month_risk'] * val_data['lat_risk']

num_val_fires = max(int(val_data.shape[0] * 0.003), 100)
val_prob = val_data['combined_risk'] / val_data['combined_risk'].sum()

val_fire_indices = np.random.choice(
    val_data.index,
    size=min(num_val_fires, len(val_data)),
    replace=False,
    p=val_prob
)
val_data.loc[val_fire_indices, 'fire_occurred'] = 1

# Prepare features and target
def prepare_model_data(df):
    y = df['fire_occurred'].values
    exclude_cols = ['fire_occurred', 'date', 'grid_cell_id', 'latitude', 'month_risk', 'lat_risk', 'combined_risk']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    
    return X, y, feature_cols

# Prepare training and validation
X_train, y_train, feature_cols = prepare_model_data(train_data)
X_val, y_val, _ = prepare_model_data(val_data)

# Train the Random Forest
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)
print("Model training complete!")

# Evaluate
print("\nEvaluating model...")
y_pred = rf_model.predict(X_val)
y_pred_proba = rf_model.predict_proba(X_val)[:, 1]

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save the model and feature columns
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_model, os.path.join(MODEL_DIR, 'wildfire_prediction_model.joblib'))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_columns.joblib'))

print(f"Model and feature columns saved to {MODEL_DIR}")

print("\n✅ Step 1 (train_model.py) update complete!")
