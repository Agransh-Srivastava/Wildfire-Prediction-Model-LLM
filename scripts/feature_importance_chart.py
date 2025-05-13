import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Load the model and feature columns
model_path = os.path.join(MODEL_DIR, 'wildfire_prediction_model.joblib')
features_path = os.path.join(MODEL_DIR, 'feature_columns.joblib')

print("Loading model and features...")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for sorting
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
})

# Sort by importance descending
importance_df = importance_df.sort_values('Importance', ascending=False)

# Get Top 20 features
top_20 = importance_df.head(20)

# Plot
plt.figure(figsize=(10, 8))
plt.barh(top_20['Feature'], top_20['Importance'], align='center')
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances for Wildfire Prediction Model')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the chart
output_path = os.path.join(MODEL_DIR, 'feature_importance_chart.png')
plt.savefig(output_path)

print(f"✅ Feature importance chart saved to: {output_path}")
