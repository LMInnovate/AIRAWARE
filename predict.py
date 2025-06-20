# src/predict.py

import joblib
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("models/ridge_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# Sample input (customizable or CLI args)
# Example values: pm2_5, pm10, no2, o3, temperature, humidity, population_density (as numeric)
# Example: 44.0 60.0 30.0 40.0 25.0 55.0 1 (Urban)
if len(sys.argv) != 8:
    print("Usage: python3 src/predict.py <pm2_5> <pm10> <no2> <o3> <temperature> <humidity> <population_density>")
    sys.exit(1)

features = np.array([[float(val) for val in sys.argv[1:]]])
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)
print(f"\n Predicted hospital admissions: {prediction[0]:.2f}")
