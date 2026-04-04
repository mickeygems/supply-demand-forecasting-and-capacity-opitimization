import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def check_performance(recent_data_path="data/forecast_output.csv", baseline_rmse=None):
    """
    Computes RMSE and MAE on the recent batch prediction where actuals are available.
    Triggers an alert if RMSE degrades significantly compared to baseline.
    """
    if not os.path.exists(recent_data_path):
        print(f"File not found: {recent_data_path}")
        return
        
    df = pd.read_csv(recent_data_path)
    
    # We can only monitor if both actual 'demand_units' and 'predicted_demand' exist
    if "demand_units" not in df.columns or "predicted_demand" not in df.columns:
        print("Missing required columns for monitoring (need 'demand_units' and 'predicted_demand').")
        return
        
    actuals = df["demand_units"]
    preds = df["predicted_demand"]
    
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    
    print(f"Current Batch RMSE: {rmse:.4f}")
    print(f"Current Batch MAE: {mae:.4f}")
    
    # Check degradation
    if baseline_rmse is not None:
        threshold = baseline_rmse * 1.30  # 30% increase
        if rmse > threshold:
            print(f"ALERT: RMSE has degraded by >30% (Baseline: {baseline_rmse:.4f}, Current: {rmse:.4f})")
            print("Action: Trigger retraining pipeline.")
            return True # Needs retraining
            
    return False

if __name__ == "__main__":
    # Example baseline RMSE from initial training
    # Ideally, this should be fetched from a central meta-store or registry
    check_performance(baseline_rmse=2000.0) 
