import streamlit as st
import pandas as st_pd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys

# Add root directory to sys path so relative imports work if started from demandcapacity folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.batch_predict import run_batch_prediction

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
st.set_page_config(page_title="Supply Demand Forecasting Dashboard", layout="wide", page_icon="☁️")

st.markdown("""
<style>

:root {
    --bg-main: #0b1220;
    --card-bg: #111827;
    --accent: #3b82f6;
    --text-primary: #e5e7eb;
    --text-muted: #9ca3af;
    --success: #22c55e;
    --warning: #f59e0b;
}

/* App background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
    color: var(--text-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #1f2937;
}

/* KPI Cards */
.kpi-card {
    background: var(--card-bg);
    border-radius: 14px;
    padding: 18px;
    border: 1px solid #1f2937;
    transition: 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent);
}

/* KPI Title */
.kpi-title {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
}

/* KPI Value */
.kpi-value {
    font-size: 28px;
    font-weight: 600;
    color: white;
}

/* Header Banner */
.header-box {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    padding: 22px;
    border-radius: 14px;
    color: white;
    margin-bottom: 20px;
}

/* Alert */
.alert-box {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 14px;
    border-radius: 10px;
    margin-top: 10px;
}

/* Tabs */
.stTabs [role="tab"] {
    background-color: transparent;
    color: var(--text-muted);
}
.stTabs [aria-selected="true"] {
    color: white;
    border-bottom: 2px solid var(--accent);
}

/* Remove Streamlit padding */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/forecast_output.csv"

# -------------------------------------------------------------
# Sidebar & Actions
# -------------------------------------------------------------
st.sidebar.title("🎛️ Navigation & Actions")

if st.sidebar.button("🔄 Trigger Batch Prediction", use_container_width=True):
    with st.spinner("Running batch prediction pipeline..."):
        try:
            run_batch_prediction(input_path="data/new_data.csv", output_path=DATA_PATH)
            st.sidebar.success("Prediction complete!")
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")

# Load Data
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

if df is None:
    st.title("Welcome to Azure Demand Forecasting")
    st.info("No forecast data found. Please place `new_data.csv` in `data/` and click 'Trigger Batch Prediction' in the sidebar.")
    st.stop()

# Ensure we have required columns
req_cols = ["date", "region", "service_type", "demand_units", "predicted_demand", "capacity_allocated"]
for c in req_cols:
    if c not in df.columns:
        if c == 'predicted_demand':
            df['predicted_demand'] = df['demand_units'] # Fallback
        elif c == 'capacity_allocated':
            df['capacity_allocated'] = df['demand_units'] * 1.2

# -------------------------------------------------------------
# Filters
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Data Filters")

regions = ["All"] + list(df["region"].unique())
services = ["All"] + list(df["service_type"].unique())

sel_region = st.sidebar.selectbox("Region", regions)
sel_service = st.sidebar.selectbox("Service Type", services)

min_date = df["date"].min()
max_date = df["date"].max()
sel_dates = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Apply filters
filtered_df = df.copy()
if sel_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == sel_region]
if sel_service != "All":
    filtered_df = filtered_df[filtered_df["service_type"] == sel_service]
if len(sel_dates) == 2:
    filtered_df = filtered_df[(filtered_df["date"] >= pd.to_datetime(sel_dates[0])) & 
                              (filtered_df["date"] <= pd.to_datetime(sel_dates[1]))]

col_logo, col_title = st.columns([1, 15])
with col_logo:
    st.image("dashboard/logo.png", width=60)
with col_title:
    st.title("Supply Demand Forecasting Dashboard")

# -------------------------------------------------------------
# Alert System
# -------------------------------------------------------------
# Check if predicted demand > 80% capacity
filtered_df["utilization"] = filtered_df["predicted_demand"] / filtered_df["capacity_allocated"]
high_util_cases = filtered_df[filtered_df["utilization"] > 0.8]

if len(high_util_cases) > 0:
    st.markdown(f'<div class="alert-box">⚠️ WARNING: {len(high_util_cases)} instances found where forecast exceeds 80% of provisioned capacity!</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# KPIs & Dashboard Layout
# -------------------------------------------------------------
def directional_accuracy(actual, pred):
    if len(actual) < 2: return 0.0
    act_diff = np.sign(np.diff(actual))
    pred_diff = np.sign(np.diff(pred))
    return np.mean(act_diff == pred_diff) * 100

valid_metrics_df = filtered_df.dropna(subset=["demand_units", "predicted_demand"])
if len(valid_metrics_df) > 0:
    rmse = np.sqrt(mean_squared_error(valid_metrics_df["demand_units"], valid_metrics_df["predicted_demand"]))
    mae = mean_absolute_error(valid_metrics_df["demand_units"], valid_metrics_df["predicted_demand"])
    dir_acc = directional_accuracy(valid_metrics_df["demand_units"].values, valid_metrics_df["predicted_demand"].values)
    # MAPE protection against dividing by zero
    nonzero_actuals = np.where(valid_metrics_df["demand_units"] == 0, 1e-9, valid_metrics_df["demand_units"])
    mape = np.mean(np.abs((valid_metrics_df["demand_units"] - valid_metrics_df["predicted_demand"]) / nonzero_actuals)) * 100
else:
    rmse, mae, dir_acc, mape = 0.0, 0.0, 0.0, 0.0

total_forecast = filtered_df["predicted_demand"].sum()
total_actual = valid_metrics_df["demand_units"].sum() if len(valid_metrics_df) > 0 else 0

peak_f_idx = filtered_df["predicted_demand"].idxmax() if len(filtered_df) > 0 and not filtered_df["predicted_demand"].isna().all() else None
peak_forecast_date = filtered_df.loc[peak_f_idx, "date"].strftime("%Y-%m-%d") if pd.notna(peak_f_idx) else "N/A"

if len(valid_metrics_df) > 0 and not valid_metrics_df["demand_units"].isna().all():
    peak_a_idx = valid_metrics_df["demand_units"].idxmax()
    peak_actual_date = valid_metrics_df.loc[peak_a_idx, "date"].strftime("%Y-%m-%d") if pd.notna(peak_a_idx) else "N/A"
else:
    peak_actual_date = "N/A"

# -------------------------------------------------------------
# Tabs layout
# -------------------------------------------------------------
tabs = st.tabs([
    "📈 Actual vs Forecast", 
    "🌍 Regional Analysis", 
    "⚙️ Service Breakdown", 
    "🔍 Monitoring & Drift", 
    "🗃️ Data Explorer", 
    "💰 Cost & Availability", 
    "📊 External Indicators"
])

# Utility for metric styling is built-in with st.metric(delta)!
# Tab 1: Actual vs Forecast
with tabs[0]:
    st.markdown("### Model Performance: Prediction vs Reality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Forecasted Demand", f"{total_forecast:,.0f}", help="Sum of predicted demand")
    c2.metric("Total Actual Demand", f"{total_actual:,.0f}", help="Sum of actual historical demand units (where available)")
    c3.metric("Forecast Accuracy (RMSE)", f"{rmse:.2f}", delta=f"{-rmse:.1f} (Lower = Better)", delta_color="inverse", help="Root Mean Squared Error. Lower is better.")
    c4.metric("MAPE", f"{mape:.2f}%", delta=f"{-mape:.1f}%", delta_color="inverse", help="Mean Absolute Percentage Error.")
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Peak Forecast Date", peak_forecast_date)
    c6.metric("Peak Actual Date", peak_actual_date)
    avg_res = (valid_metrics_df["demand_units"] - valid_metrics_df["predicted_demand"]).mean() if len(valid_metrics_df)>0 else 0
    max_res = (valid_metrics_df["demand_units"] - valid_metrics_df["predicted_demand"]).abs().max() if len(valid_metrics_df)>0 else 0
    c7.metric("Avg Residual (Actual - Pred)", f"{avg_res:+.2f}")
    c8.metric("Max Residual (Abs)", f"{max_res:,.2f}")

    st.markdown("---")
    colA, colB = st.columns([2, 1])
    with colA:
        agg_df = filtered_df.groupby("date")[["demand_units", "predicted_demand"]].sum().reset_index()
        fig1 = px.line(agg_df, x="date", y=["demand_units", "predicted_demand"], 
                       labels={"value": "Demand Units", "variable": "Type"},
                       color_discrete_sequence=["#a0a0b0", "#00d2ff"],
                       title="Actual vs Forecast Demand Over Time")
        fig1.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        filtered_df["residuals"] = filtered_df["demand_units"] - filtered_df["predicted_demand"]
        fig2 = px.histogram(filtered_df, x="residuals", nbins=50, color="service_type",
                            title="Residuals Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2: Regional Analysis
with tabs[1]:
    st.markdown("### Demand by Geography")
    
    reg_df = filtered_df.groupby("region").agg({
        "predicted_demand": "sum",
        "demand_units": "sum",
        "capacity_allocated": "sum"
    }).reset_index()
    reg_df["utilization"] = (reg_df["demand_units"] / reg_df["capacity_allocated"]) * 100
    reg_df["utilization"] = reg_df["utilization"].fillna(0)
    
    # Variance & RMSE requires grouped calc
    reg_rmse = {}
    reg_var = {}
    for r in reg_df["region"]:
        r_df = valid_metrics_df[valid_metrics_df["region"] == r]
        if len(r_df) > 0:
            reg_rmse[r] = np.sqrt(mean_squared_error(r_df["demand_units"], r_df["predicted_demand"]))
            reg_var[r] = r_df["demand_units"].std()
        else:
            reg_rmse[r] = 0
            reg_var[r] = 0
            
    # Try finding highest growth by comparing to previous period.
    try:
        fh = filtered_df.iloc[:len(filtered_df)//2].groupby("region")["predicted_demand"].sum()
        sh = filtered_df.iloc[len(filtered_df)//2:].groupby("region")["predicted_demand"].sum()
        reg_growth = ((sh - fh) / fh * 100).fillna(0)
        highest_growth_reg = reg_growth.idxmax() if len(reg_growth) > 0 else "N/A"
        highest_growth_val = reg_growth.max() if len(reg_growth) > 0 else 0
    except:
        highest_growth_reg, highest_growth_val = "N/A", 0

    lowest_util_idx = reg_df["utilization"].idxmin() if len(reg_df)>0 else None
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest Forecast Growth Region", str(highest_growth_reg).title(), f"{highest_growth_val:+.1f}%")
    if lowest_util_idx is not None:
        c2.metric("Lowest Capacity Utilization", str(reg_df.loc[lowest_util_idx, "region"]).title(), f"{reg_df.loc[lowest_util_idx, 'utilization']:.1f}%")
    else:
        c2.metric("Lowest Capacity Utilization", "N/A")
    c3.metric("Avg Regional Variance", f"{np.mean(list(reg_var.values())):.1f}" if reg_var else "0")
    c4.metric("Avg Regional RMSE", f"{np.mean(list(reg_rmse.values())):.1f}" if reg_rmse else "0")
    
    colA, colB = st.columns(2)
    with colA:
        fig3 = px.bar(filtered_df.groupby(["region", "service_type"])["predicted_demand"].sum().reset_index(), 
                      x="region", y="predicted_demand", color="service_type",
                      title="Forecast by Region (Stacked)", text_auto='.2s',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)
    with colB:
        heat_df = filtered_df.groupby(["region", "service_type"])["utilization"].max().reset_index()
        heat_pivot = heat_df.pivot(index="service_type", columns="region", values="utilization")
        fig4 = px.imshow(heat_pivot, text_auto=".1%", aspect="auto", 
                         title="Max Capacity Utilization Heatmap",
                         color_continuous_scale="RdYlGn_r", zmin=0.5, zmax=1.0)
        fig4.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

# Tab 3: Service Breakdown
with tabs[2]:
    st.markdown("### Differences by Service Type")
    
    srv_df = filtered_df.groupby("service_type").agg({
        "predicted_demand": "sum",
        "demand_units": "sum",
        "capacity_allocated": "sum",
        "cost_usd": "sum"
    }).reset_index()
    srv_df["utilization"] = (srv_df["demand_units"] / srv_df["capacity_allocated"] * 100).fillna(0)
    srv_df["perc_diff"] = ((srv_df["predicted_demand"] - srv_df["demand_units"]) / srv_df["demand_units"] * 100).fillna(0)

    for i, row in srv_df.iterrows():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"[{row['service_type']}] Forecast", f"{row['predicted_demand']:,.0f}")
        c2.metric(f"Actual vs Forecast % Diff", f"{row['perc_diff']:+.1f}%")
        c3.metric(f"Capacity Utilization", f"{row['utilization']:.1f}%")
        c4.metric(f"Total Cost USD", f"${row['cost_usd']:,.2f}")
        st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        fig5 = px.pie(srv_df, names="service_type", values="predicted_demand", hole=0.4,
                      title="Forecast Distribution by Service", color_discrete_sequence=px.colors.qualitative.Set3)
        fig5.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)
    with colB:
        fig6 = px.bar(srv_df, x="service_type", y=["cost_usd"], title="Cost per Service", text_auto='.2s', color="service_type", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig6.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

# Tab 4: Monitoring & Drift
with tabs[3]:
    st.markdown("### Model Performance and Anomalies")
    
    # Calculate rolling MAE for drift
    if len(valid_metrics_df) > 0:
        drift_df = valid_metrics_df.groupby("date").agg({"demand_units":"sum", "predicted_demand":"sum"}).reset_index()
        drift_df["error"] = np.abs(drift_df["demand_units"] - drift_df["predicted_demand"])
        drift_df["rolling_mae"] = drift_df["error"].rolling(window=7, min_periods=1).mean()
        
        # Outliers > 2 std dev
        std_resid = filtered_df["residuals"].std()
        mean_resid = filtered_df["residuals"].mean()
        filtered_df["is_outlier"] = np.abs(filtered_df["residuals"] - mean_resid) > (2 * std_resid)
        outlier_count = filtered_df["is_outlier"].sum()
    else:
        outlier_count = 0

    c1, c2, c3, c4 = st.columns(4)
    current_mae = drift_df["rolling_mae"].iloc[-1] if len(valid_metrics_df) > 0 else 0
    start_mae = drift_df["rolling_mae"].iloc[0] if len(valid_metrics_df) > 0 else 0
    drift_trend = current_mae - start_mae
    
    c1.metric("Current Rolling MAE (7d)", f"{current_mae:.2f}", delta=f"{drift_trend:+.2f} Drift", delta_color="inverse", help="Tracks model accuracy drift over time.")
    c2.metric("Capacity Alerts", f"{len(high_util_cases)}", help="Forecast > 80% Capacity")
    c3.metric("Anomalies (>2 StdDev)", f"{outlier_count}", help="Days where prediction error is unusually large.")
    c4.metric("Directional Accuracy", f"{dir_acc:.1f}%", help="% of time forecast predicts correct up/down movement.")

    colA, colB = st.columns(2)
    with colA:
        if len(valid_metrics_df) > 0:
            fig7 = px.line(drift_df, x="date", y="rolling_mae", title="Model Drift: 7-Day Rolling MAE")
            fig7.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig7, use_container_width=True)
    with colB:
        if len(valid_metrics_df) > 0:
            fig8 = px.scatter(filtered_df, x="date", y="residuals", color="is_outlier", 
                              title="Prediction Anomalies", color_discrete_map={True: '#ff4b4b', False: '#1e88e5'})
            fig8.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig8, use_container_width=True)

# Tab 5: Data Explorer
with tabs[4]:
    st.markdown("### Raw and Filtered Data View")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Displayed", len(filtered_df))
    avg_demand = filtered_df["demand_units"].mean()
    c2.metric("Avg Actual Demand", f"{avg_demand:,.1f}" if pd.notna(avg_demand) else "N/A")
    c3.metric("Avg Capacity Utilization", f"{reg_df['utilization'].mean():.1f}%" if len(reg_df)>0 else "N/A")
    c4.metric("Avg Daily Cost", f"${filtered_df['cost_usd'].mean():.2f}" if "cost_usd" in filtered_df else "N/A")

    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data as CSV", data=csv, file_name="forecast_filtered.csv", mime="text/csv")

# Tab 6: Cost & Availability
with tabs[5]:
    st.markdown("### Financial and Operational Insights")
    
    total_cost = filtered_df["cost_usd"].sum() if "cost_usd" in filtered_df else 0
    avg_cost_unit = total_cost / total_actual if total_actual > 0 else 0
    avg_avail = filtered_df["availability"].mean() if "availability" in filtered_df else 0
    
    if "availability" in filtered_df:
        avail_df = filtered_df.groupby("service_type")["availability"].mean().reset_index()
        lowest_avail_svc = avail_df.loc[avail_df["availability"].idxmin(), "service_type"] if len(avail_df)>0 else "N/A"
        lowest_avail_val = avail_df["availability"].min() if len(avail_df)>0 else 0
    else:
        lowest_avail_svc, lowest_avail_val = "N/A", 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost (USD)", f"${total_cost:,.2f}")
    c2.metric("Avg Cost per Demand Unit", f"${avg_cost_unit:.3f}")
    c3.metric("Avg Availability", f"{avg_avail:.2f}%")
    c4.metric("Lowest Availability Service", f"{lowest_avail_svc}", f"{lowest_avail_val-100:.2f}%" if lowest_avail_svc != "N/A" else "0%")

    colA, colB = st.columns(2)
    with colA:
        if "cost_usd" in filtered_df and "capacity_allocated" in filtered_df:
            fig9 = px.scatter(filtered_df, x="capacity_allocated", y="cost_usd", color="service_type", title="Cost vs Capacity Provisioned")
            fig9.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig9, use_container_width=True)
    with colB:
        if "availability" in filtered_df:
            avail_trend = filtered_df.groupby("date")["availability"].mean().reset_index()
            fig10 = px.line(avail_trend, x="date", y="availability", title="Availability Over Time")
            fig10.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig10.update_yaxes(range=[min(99.0, avail_trend["availability"].min()), 100.1])
            st.plotly_chart(fig10, use_container_width=True)

# Tab 7: External Indicators
with tabs[6]:
    st.markdown("### External Business Drivers")
    c1, c2, c3, c4 = st.columns(4)
    
    mkt = filtered_df["market_demand_index"].mean() if "market_demand_index" in filtered_df else 0
    gdp = filtered_df["gdp_growth"].mean() if "gdp_growth" in filtered_df else 0
    cgr = filtered_df["customer_growth_rate"].mean() if "customer_growth_rate" in filtered_df else 0
    pev = filtered_df["pricing_event"].sum() if "pricing_event" in filtered_df else 0
    
    c1.metric("Avg Market Demand Index", f"{mkt:.1f}")
    c2.metric("Avg GDP Growth", f"{gdp:.2f}%")
    c3.metric("Avg Customer Growth", f"{cgr:.2f}%")
    c4.metric("Total Pricing Events", f"{pev:,.0f}")

    colA, colB = st.columns(2)
    with colA:
        if "market_demand_index" in filtered_df:
            fig11 = px.scatter(filtered_df, x="market_demand_index", y="predicted_demand", trendline="ols", title="Market Index vs Forecast")
            fig11.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig11, use_container_width=True)
    with colB:
        if "gdp_growth" in filtered_df:
            fig12 = px.scatter(filtered_df, x="gdp_growth", y="predicted_demand", trendline="ols", title="GDP Growth vs Forecast")
            fig12.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig12, use_container_width=True)

