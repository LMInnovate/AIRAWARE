import sys
import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/ridge_model.joblib')  # Use the correct path and model name

# Collect input features from command-line arguments
if len(sys.argv) != 8:
    print("Usage: python predict.py <PM2.5> <PM10> <NO2> <SO2> <O3> <CO> <is_urban>")
    sys.exit(1)

try:
    features = list(map(float, sys.argv[1:]))
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    print(f"Predicted hospital admissions: {prediction[0]:.2f}")
except ValueError:
    print("Invalid input: All inputs must be numbers.")
