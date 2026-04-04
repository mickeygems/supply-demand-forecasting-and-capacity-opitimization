import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pipeline.preprocess import full_preprocess

def train_and_save(data_path="azure_demand_uncleaned_10k.csv", model_dir="models"):
    """
    Train the XGBoost model mimicking the notebook logic but using proper encoding
    for categorical features, then save it to disk.
    """
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please provide correct path.")
        return

    print("Preprocessing data (cleaning, feature engineering, encoding)...")
    # full_preprocess drops 'date' column implicitly? 
    # The notebook kept 'date' until right before training.
    # Let's ensure 'date' is kept for train/test split.
    os.makedirs(model_dir, exist_ok=True)
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    
    processed_df = full_preprocess(df, is_training=True, encoder_path=encoder_path)
    
    # Drop rows with NaNs caused by lags
    processed_df = processed_df.dropna().reset_index(drop=True)
    
    print(f"Data shape after preprocessing: {processed_df.shape}")
    
    # Ensure date is sorted
    processed_df['date'] = pd.to_datetime(processed_df['date'])
    processed_df = processed_df.sort_values('date')
    
    # Train/Test Split (80/20 chronological split from notebook)
    split_idx = int(len(processed_df) * 0.8)
    
    train = processed_df.iloc[:split_idx]
    test = processed_df.iloc[split_idx:]
    
    # Features to drop before training
    # We no longer drop region/service_type because they are OneHotEncoded!
    drop_cols = ["date", "demand_units"]
    
    X_train = train.drop(columns=drop_cols)
    y_train = train["demand_units"]
    
    X_test = test.drop(columns=drop_cols)
    y_test = test["demand_units"]
    
    print("\nTraining XGBoost model...")
    # Best parameters from notebook GridSearch:
    # We'll use these directly to speed up training, or close to them.
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,      # From best_params
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    
    print("\nEvaluating model...")
    preds = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"XGBoost MAE: {mae:.4f}")
    print(f"XGBoost RMSE: {rmse:.4f}")
    print(f"XGBoost R2: {r2:.4f}")
    
    # Save model and feature names
    model_path = os.path.join(model_dir, "xgb_model.pkl")
    joblib.dump(xgb_model, model_path)
    print(f"\nModel saved successfully to {model_path}")
    
    # Save feature names to ensure API inputs align correctly
    features_path = os.path.join(model_dir, "feature_names.pkl")
    joblib.dump(list(X_train.columns), features_path)
    print(f"Feature names saved directly to {features_path}")

if __name__ == "__main__":
    import sys
    # For local execution if run directly
    data_file = sys.argv[1] if len(sys.argv) > 1 else "../azure_demand_uncleaned_10k.csv"
    if not os.path.exists(data_file):
        data_file = "azure_demand_uncleaned_10k.csv"
    train_and_save(data_file)
