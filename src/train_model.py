# src/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load the data
DATA_PATH = "data/air_quality_health_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Handle missing values (if any)
df = df.dropna()

# Encode categorical variables
df['population_density'] = df['population_density'].astype('category').cat.codes

# Features and Target
features = ['pm2_5', 'pm10', 'no2', 'o3', 'temperature', 'humidity', 'population_density']
target = 'hospital_admissions'

X = df[features]
y = df[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Try three models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    avg_cv_r2 = np.mean(cv_scores)

    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print(f"Average CV R2: {avg_cv_r2:.2f}")

    # Save best model (just as example: save last one)
    joblib.dump(model, f"models/{name.lower()}_model.joblib")
