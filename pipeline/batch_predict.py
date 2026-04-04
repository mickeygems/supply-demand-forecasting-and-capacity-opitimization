import pandas as pd
import joblib
import os
from pipeline.preprocess import full_preprocess

def run_batch_prediction(input_path="data/new_data.csv", output_path="data/forecast_output.csv"):
    """
    Reads new data, preprocesses it, predicts using XGBoost, and saves results.
    """
    model_dir = "models"
    model_path = os.path.join(model_dir, "xgb_model.pkl")
    features_path = os.path.join(model_dir, "feature_names.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    
    if not os.path.exists(input_path):
        print(f"No file found at {input_path}")
        return
        
    print(f"Loading incoming batch data from {input_path}...")
    df = pd.read_csv(input_path)
    
    try:
        model = joblib.load(model_path)
        encoded_feature_names = joblib.load(features_path)
    except FileNotFoundError:
        print("Model or feature names not found. Please train first.")
        return

    print("Preprocessing data...")
    # For batch prediction, we use is_training=False so the encoder is only transformed
    processed_df = full_preprocess(df, is_training=False, encoder_path=encoder_path)
    
    # Store original data for final output
    output_df = df.copy()
    
    # Drop target and date from features
    if "date" in processed_df.columns:
        processed_df = processed_df.drop(columns=["date"])
    if "demand_units" in processed_df.columns:
        processed_df = processed_df.drop(columns=["demand_units"])
        
    # Ensure exact column match with training data
    for col in encoded_feature_names:
        if col not in processed_df.columns:
            processed_df[col] = 0.0
            
    # Reorder
    X = processed_df[encoded_feature_names]
    
    print("Generating predictions...")
    preds = model.predict(X)
    
    output_df["predicted_demand"] = preds
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Batch prediction completed. Results saved to {output_path}")

if __name__ == "__main__":
    run_batch_prediction()
