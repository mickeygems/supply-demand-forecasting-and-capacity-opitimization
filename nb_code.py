# --- Cell 1 ---
import pandas as pd
import numpy as np

# --- Cell 2 ---
#LOAD DATA
df = pd.read_csv("azure_demand_uncleaned_10k.csv")

print("Initial Shape:", df.shape)
print(df.head())


# --- Cell 3 ---

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Drop empty daterows
df = df.dropna(subset=["date"])


# --- Cell 4 ---
#REMOVE DUPLICATES
df = df.drop_duplicates()


# --- Cell 5 ---
# HANDLE MISSING VALUES

# Fill demand using region+service median
df["demand_units"] = df.groupby(
    ["region", "service_type"]
)["demand_units"].transform(
    lambda x: x.fillna(x.median())
)
# Fill capacity using 1.25 × demand if missing
df["capacity_allocated"] = df["capacity_allocated"].fillna(
    df["demand_units"] * 1.25
)

# cost using demand × price logic
compute_price = 0.085
storage_price = 0.022

def fill_cost(row):
    if pd.isna(row["cost_usd"]):
        if row["service_type"] == "Compute":
            return row["demand_units"] * compute_price
        else:
            return row["demand_units"] * storage_price
    return row["cost_usd"]

df["cost_usd"] = df.apply(fill_cost, axis=1)


# --- Cell 6 ---
# FIX INVALID AVAILABILITY

# Remove unrealistic SLA (>100%)
df.loc[df["availability"] > 100, "availability"] = 99.99

# Clip to realistic SLA range
df["availability"] = df["availability"].clip(99.0, 100.0)

# --- Cell 7 ---
#HANDLE OUTLIERS (IQR METHOD)

def remove_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[column].clip(lower, upper)

df["demand_units"] = remove_outliers("demand_units")
df["capacity_allocated"] = remove_outliers("capacity_allocated")



# --- Cell 8 ---

# FIX DATA TYPES

numeric_columns = [
    "demand_units",
    "capacity_allocated",
    "cost_usd",
    "availability",
    "market_demand_index",
    "gdp_growth",
    "customer_growth_rate",
    "industry_mix_index"
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Cell 9 ---
# SORT DATA
df = df.sort_values(by=["region", "service_type", "date"])

# --- Cell 10 ---
print("\nAfter Cleaning Shape:", df.shape)
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# SAVE CLEAN DATA

df.to_csv("azure_demand_cleaned.csv", index=False)

print("\n✅ Cleaned dataset saved as 'azure_demand_cleaned.csv'")

# --- Cell 12 ---
import pandas as pd
import numpy as np

df = pd.read_csv("azure_demand_cleaned.csv")

df["date"] = pd.to_datetime(df["date"])

print("Shape:", df.shape)


print("\n🔎 Missing Values Check:")
print(df.isnull().sum())



print("\n🔎 Time Continuity Check:")

time_issues = []

for (region, service), group in df.groupby(["region", "service_type"]):
    group = group.sort_values("date")
    date_diff = group["date"].diff().dropna()

    # Check if gaps > 1 day exist
    if (date_diff > pd.Timedelta(days=1)).any():
        time_issues.append((region, service))

if len(time_issues) == 0:
    print("✅ No time gaps found.")
else:
    print("⚠ Time gaps found in:")
    for issue in time_issues:
        print(issue)


print("\n🔎 Demand Distribution:")
print(df["demand_units"].describe())

print("\n🔎 Capacity Distribution:")
print(df["capacity_allocated"].describe())


# --- Cell 13 ---
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter

# --- Cell 14 ---
import pandas as pd

# Reload the cleaned data to ensure 'region' and 'service_type' columns are present
df = pd.read_csv("azure_demand_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])

# Re-create date-related features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter

print("DataFrame reloaded and date features recreated.")

# --- Cell 15 ---
df["lag_1"] = df.groupby(
    ["region", "service_type"]
)["demand_units"].shift(1)

df["lag_7"] = df.groupby(
    ["region", "service_type"]
)["demand_units"].shift(7)

df["rolling_mean_7"] = df.groupby(
    ["region", "service_type"]
)["demand_units"].transform(lambda x: x.rolling(7).mean())


# --- Cell 16 ---
split_date = df["date"].quantile(0.8)

train = df[df["date"] <= split_date]
test = df[df["date"] > split_date]

# --- Cell 17 ---
df = df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"])

# --- Cell 18 ---
print(df.head())

# --- Cell 20 ---
import numpy as np
import pandas as pd

# Ensure datetime format
df['date'] = pd.to_datetime(df['date'])

# Sort properly to avoid leakage
df = df.sort_values(['region', 'service_type', 'date'])

# Reset index
df = df.reset_index(drop=True)

# --- Cell 21 ---
# Extracting time-related features from the 'date' column
df['month'] = df['date'].dt.month  # Extracts month (1-12)
df['quarter'] = df['date'].dt.quarter  # Extracts quarter (1-4)
df['day_of_week'] = df['date'].dt.dayofweek  # Extracts day of week (0=Monday, 6=Sunday)
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Flags weekends (1 for Saturday/Sunday, 0 for weekdays)

print(df.head())


# --- Cell 22 ---
# Creating lag features to capture temporal dependencies
df['lag_1'] = df.groupby(['region', 'service_type'])['demand_units'].shift(1)
df['lag_7'] = df.groupby(['region', 'service_type'])['demand_units'].shift(7)
df['lag_30'] = df.groupby(['region', 'service_type'])['demand_units'].shift(30)  # 30-day lag
df['lag_60'] = df.groupby(['region', 'service_type'])['demand_units'].shift(60)  # 60-day lag

print(df[['date', 'region', 'service_type', 'demand_units','lag_1', 'lag_7', 'lag_30', 'lag_60']].head())


# --- Cell 23 ---
# Creating rolling mean features
df['rolling_mean_7'] = df.groupby(['region', 'service_type'])['demand_units'].transform(lambda x: x.rolling(7).mean())  # 7-day rolling mean
df['rolling_mean_30'] = df.groupby(['region', 'service_type'])['demand_units'].transform(lambda x: x.rolling(30).mean())  # 30-day rolling mean

print(df[['date', 'region', 'service_type', 'demand_units', 'rolling_mean_7', 'rolling_mean_30']].head())


# --- Cell 24 ---
# Create seasonal flags (e.g., is summer, is holiday period)
df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)  # Flags summer months (June, July, August)
df['is_holiday_period'] = df['month'].isin([11, 12]).astype(int)  # Flags holiday period (November, December)

print(df[['date', 'is_summer', 'is_holiday_period']].head())


# --- Cell 25 ---
df['previous_year_demand'] = df.groupby(['region', 'service_type'])['demand_units'].shift(365)  # Assuming daily data and 365 days in a year
df['yoy_growth'] = np.where(
    df['previous_year_demand'] > 0,
    (df['demand_units'] - df['previous_year_demand']) /
    df['previous_year_demand'] * 100,
    0
)  # Calculate YoY growth in percentage

print(df[['date', 'region', 'service_type', 'demand_units', 'previous_year_demand', 'yoy_growth']].head())


# --- Cell 26 ---
#interaction features between different economic variables
df['economic_growth_interaction'] = df['gdp_growth'] * df['customer_growth_rate'] 

print(df[['date', 'region', 'service_type', 'gdp_growth', 'customer_growth_rate', 'economic_growth_interaction']].head())


# --- Cell 27 ---
df['time_index'] = (
    df.groupby(['region', 'service_type'])
      .cumcount()
)

# --- Cell 28 ---
# Fill missing values in the new features with forward fill or interpolation
external_cols = ['gdp_growth', 'customer_growth_rate']

df[external_cols] = (
    df.groupby(['region', 'service_type'])[external_cols]
      .ffill()
)

# Drop rows created due to lag/rolling
df = df.dropna().reset_index(drop=True)
# Ensure no missing values remain
print(f"Missing values after feature engineering: \n{df.isnull().sum()}")




# --- Cell 29 ---
df.to_csv("azure_demand_enriched.csv", index=False)
print("Milestone 2 dataset saved successfully.")

# --- Cell 31 ---
# !pip install statsmodels


# --- Cell 32 ---
# !pip install xgboost

# --- Cell 33 ---
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Cell 34 ---
df = pd.read_csv("azure_demand_enriched.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print(df.head())

# --- Cell 35 ---
#timeseries arima
ts_df = df[(df['region'] == "centralus") & (df['service_type'] == "Compute")]

ts_df = ts_df.set_index("date")

y = ts_df['demand_units']

# --- Cell 36 ---
#training and splitting data
train_size = int(len(ts_df) * 0.8)

train = ts_df.iloc[:train_size]
test = ts_df.iloc[train_size:]

y_train = train['demand_units']
y_test = test['demand_units']

# --- Cell 37 ---
#arima model
model_arima = ARIMA(y_train, order=(2,1,2))
arima_fit = model_arima.fit()

print(arima_fit.summary())

# --- Cell 38 ---
#forecasting
arima_pred = arima_fit.forecast(steps=len(y_test))

arima_pred = pd.Series(arima_pred, index=y_test.index)

#evaluation
mae_arima = mean_absolute_error(y_test, arima_pred)
rmse_arima = np.sqrt(mean_squared_error(y_test, arima_pred))

print("ARIMA MAE:", mae_arima)
print("ARIMA RMSE:", rmse_arima)

# --- Cell 39 ---
#preparing data for XGboost and splitting
features = df.drop(columns=[
    "date",
    "region",
    "service_type",
    "demand_units"
])

target = df["demand_units"]

#Train and test split
split = int(len(df)*0.8)

X_train = features.iloc[:split]
X_test = features.iloc[split:]

y_train = target.iloc[:split]
y_test = target.iloc[split:]

# --- Cell 40 ---
#XGBoost model
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# --- Cell 41 ---
#xgboostpredictions and evaluations
xgb_pred = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, xgb_pred)
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))
r2_xgb = r2_score(y_test, xgb_pred)

print("XGBoost MAE:", mae_xgb)
print("XGBoost RMSE:", rmse_xgb)
print("XGBoost R2:", r2_xgb)


# --- Cell 42 ---
#Hyperparameter tuning
param_grid = {
    "max_depth": [4,6,8],
    "learning_rate": [0.01,0.05,0.1],
    "n_estimators": [100,200],
    "subsample": [0.8,1]
}

grid = GridSearchCV(
    estimator=XGBRegressor(objective="reg:squarederror"),
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    verbose=1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# --- Cell 43 ---
#Training better modelwith its metrics
best_xgb = grid.best_estimator_

best_pred = best_xgb.predict(X_test)

mae = mean_absolute_error(y_test, best_pred)
rmse = np.sqrt(mean_squared_error(y_test, best_pred))
r2 = r2_score(y_test, best_pred)

print("Final XGBoost MAE:", mae)
print("Final XGBoost RMSE:", rmse)
print("Final XGBoost R2:", r2)

# --- Cell 44 ---
#forecasting and model comaparision
plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual")
plt.plot(arima_pred.values, label="ARIMA Forecast")
plt.plot(best_pred, label="XGBoost Forecast")

plt.legend()
plt.title("Demand Forecast Comparison")
plt.show()


results = pd.DataFrame({
    "Model": ["ARIMA", "XGBoost"],
    "MAE": [mae_arima, mae],
    "RMSE": [rmse_arima, rmse]
})

print(results)

# --- Cell 45 ---
import matplotlib.pyplot as plt

# Convert predictions to Series with same index
arima_pred_series = pd.Series(arima_pred, index=y_test.index)
xgb_pred_series = pd.Series(best_pred, index=y_test.index)

# Sort by date (very important)
y_test = y_test.sort_index()
arima_pred_series = arima_pred_series.sort_index()
xgb_pred_series = xgb_pred_series.sort_index()

plt.figure(figsize=(14,6))

plt.plot(y_test.index, y_test, label="Actual Demand", linewidth=2)
plt.plot(arima_pred_series.index, arima_pred_series, label="ARIMA Forecast")
plt.plot(xgb_pred_series.index, xgb_pred_series, label="XGBoost Forecast")

plt.title("Demand Forecast Comparison: Actual vs ARIMA vs XGBoost")
plt.xlabel("Date")
plt.ylabel("Demand Units")
plt.legend()
plt.grid(True)

plt.show()

# --- Cell 46 ---
plt.figure(figsize=(14,6))

plt.plot(y_test.tail(200), label="Actual")
plt.plot(arima_pred_series.tail(200), label="ARIMA")
plt.plot(xgb_pred_series.tail(200), label="XGBoost")

plt.legend()
plt.title("Model Comparison (Last 200 Days)")
plt.show()

