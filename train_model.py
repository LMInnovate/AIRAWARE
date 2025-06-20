# src/train_model.py

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import load_data, clean_data, feature_target_split

def train(filepath):
    df = load_data(filepath)
    if df is None:
        return

    df = clean_data(df)


    features = ['PM2.5', 'PM10', 'NO2', 'O3', 'CO']
    target = 'Hospital_Admissions'

    X, y = feature_target_split(df, features, target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[INFO] R² score: {r2_score(y_test, y_pred):.4f}")
    print(f"[INFO] MSE: {mean_squared_error(y_test, y_pred):.2f}")

    joblib.dump(model, 'airaware_model.pkl')
    print("[INFO] Model saved as 'airaware_model.pkl'")

if __name__ == '__main__':
    train('data/air_quality_health_dataset.csv')
