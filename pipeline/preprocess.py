import pandas as pd
import numpy as np
import joblib
import os

def clean_data(df):
    """
    Apply the data cleaning steps defined in the Jupyter Notebook.
    """
    df = df.copy()
    
    # Check if 'date' column exists
    if 'date' in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    
    df = df.drop_duplicates()
    
    # Make sure we don't crash if columns are missing during real-time API prediction
    if "demand_units" in df.columns and "region" in df.columns and "service_type" in df.columns:
        df["demand_units"] = df.groupby(["region", "service_type"])["demand_units"].transform(
            lambda x: x.fillna(x.median())
        )
    
    if "capacity_allocated" in df.columns and "demand_units" in df.columns:
        df["capacity_allocated"] = df["capacity_allocated"].fillna(df["demand_units"] * 1.25)
    
    compute_price = 0.085
    storage_price = 0.022

    def fill_cost(row):
        if pd.isna(row.get("cost_usd")):
            if row.get("service_type") == "Compute":
                return row.get("demand_units", 0) * compute_price
            else:
                return row.get("demand_units", 0) * storage_price
        return row.get("cost_usd")

    if "cost_usd" in df.columns:
        df["cost_usd"] = df.apply(fill_cost, axis=1)

    if "availability" in df.columns:
        df.loc[df["availability"] > 100, "availability"] = 99.99
        df["availability"] = df["availability"].clip(99.0, 100.0)

    # IQR outlier removal (simplified for batch/train, skip in API if single row)
    if len(df) > 10:
        def remove_outliers(column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return df[column].clip(lower, upper)

        if "demand_units" in df.columns:
            df["demand_units"] = remove_outliers("demand_units")
        if "capacity_allocated" in df.columns:
            df["capacity_allocated"] = remove_outliers("capacity_allocated")

    numeric_columns = [
        "demand_units", "capacity_allocated", "cost_usd", "availability",
        "market_demand_index", "gdp_growth", "customer_growth_rate", "industry_mix_index"
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Sort data if date exists
    if 'date' in df.columns and 'region' in df.columns and 'service_type' in df.columns:
        df = df.sort_values(by=["region", "service_type", "date"]).reset_index(drop=True)
        
    return df

def extract_time_features(df):
    """
    Extract date and time features just like the original notebook.
    """
    df = df.copy()
    if 'date' not in df.columns:
        return df

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_holiday_period'] = df['month'].isin([11, 12]).astype(int)
    
    return df

def generate_lag_features(df):
    """
    Generate lag and rolling features. Requires historical data ordered by date.
    Note: For a production API, these values are typically calculated using a 
    feature store or passed in. Here we calculate them if the DataFrame has history.
    """
    df = df.copy()
    
    # If the dataframe has only 1 row (API request without history), we might need to skip
    if len(df) <= 1:
        # Impute zeros or defaults for the API request if lag features are completely missing
        lag_cols = ['lag_1', 'lag_7', 'lag_30', 'lag_60', 'rolling_mean_7', 'rolling_mean_30', 'previous_year_demand', 'yoy_growth']
        for col in lag_cols:
            if col not in df.columns:
                df[col] = 0.0 # Simplistic imputation for isolated single predictions
        return df

    # Lags
    df['lag_1'] = df.groupby(['region', 'service_type'])['demand_units'].shift(1)
    df['lag_7'] = df.groupby(['region', 'service_type'])['demand_units'].shift(7)
    df['lag_30'] = df.groupby(['region', 'service_type'])['demand_units'].shift(30)
    df['lag_60'] = df.groupby(['region', 'service_type'])['demand_units'].shift(60)
    
    # Rolling means
    df['rolling_mean_7'] = df.groupby(['region', 'service_type'])['demand_units'].transform(lambda x: x.rolling(7).mean())
    df['rolling_mean_30'] = df.groupby(['region', 'service_type'])['demand_units'].transform(lambda x: x.rolling(30).mean())
    
    # Year-over-Year
    df['previous_year_demand'] = df.groupby(['region', 'service_type'])['demand_units'].shift(365)
    df['yoy_growth'] = np.where(
        (df['previous_year_demand'].notna()) & (df['previous_year_demand'] > 0),
        (df['demand_units'] - df['previous_year_demand']) / df['previous_year_demand'] * 100,
        0
    )
    
    # Time index
    df['time_index'] = df.groupby(['region', 'service_type']).cumcount()
    return df

def add_advanced_features(df):
    """
    Add advanced interaction features.
    """
    df = df.copy()
    if 'gdp_growth' in df.columns and 'customer_growth_rate' in df.columns:
        df['economic_growth_interaction'] = df['gdp_growth'] * df['customer_growth_rate']
    return df

def encode_categoricals(df, encoder_path="models/encoder.pkl", train_mode=False):
    """
    Encodes 'region' and 'service_type' using OneHotEncoder.
    This was an Improvement implemented on top of the original notebook.
    """
    df = df.copy()
    columns_to_encode = ['region', 'service_type']
    
    # Only try to encode if the columns exist
    if not all(col in df.columns for col in columns_to_encode):
        return df

    from sklearn.preprocessing import OneHotEncoder

    if train_mode:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[columns_to_encode])
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        joblib.dump(encoder, encoder_path)
    else:
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            encoded_data = encoder.transform(df[columns_to_encode])
        else:
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")

    # Create distinct column names for the encoded data
    encoded_cols = encoder.get_feature_names_out(columns_to_encode)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
    
    # Concatenate and drop original columns
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=columns_to_encode)
    
    return df

def full_preprocess(df, is_training=False, encoder_path="models/encoder.pkl"):
    """
    Executes the full pipeline.
    """
    df = clean_data(df)
    df = extract_time_features(df)
    df = generate_lag_features(df)
    df = add_advanced_features(df)
    
    # Forward fill missing values for specific columns
    external_cols = ['gdp_growth', 'customer_growth_rate']
    if all(col in df.columns for col in external_cols) and 'region' in df.columns and 'service_type' in df.columns:
        df[external_cols] = df.groupby(['region', 'service_type'])[external_cols].ffill()
    
    df = encode_categoricals(df, encoder_path=encoder_path, train_mode=is_training)
    
    return df
