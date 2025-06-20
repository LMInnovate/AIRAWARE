# src/preprocess.py

import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    print("[INFO] Cleaned dataset (removed nulls and duplicates)")
    return df

def feature_target_split(df, feature_cols, target_col):
    X = df[feature_cols]
    y = df[target_col]
    return X, y
