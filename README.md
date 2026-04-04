# Azure Demand Forecasting & Capacity Optimization - Production System  

LINK : https://supply-demand-forecasting-and-capacity-optimization-g9mkjxqk3j.streamlit.app/

This project transforms the Azure Demand Forecasting model into a fully functional, production-ready pipeline.

## Features

- **Training Pipeline**: Automatically recreates the XGBoost model from the notebook logic, adds categorical encoding, and serializes artifacts (`models/xgb_model.pkl`).
- **REST API**: A FastAPI application providing a real-time `/predict` endpoint.
- **Batch Processing**: Periodically scores a large set of data (`data/new_data.csv`) and generates `forecast_output.csv`.
- **Monitoring**: Checks model drift and computes RMSE/MAE metrics.
- **Dashboard**: A professional dark-mode Streamlit dashboard with KPI cards, Plotly visualizations, and capacity alerting.

---

## 🚀 How to Run Locally

### 1. Setup Environment
Ensure Python 3.10+ is installed.
```bash
pip install -r requirements.txt
```

### 2. Generate Models
Extracts the logic from the notebook, trains the model, and creates the required `models/*.pkl` files.
```bash
python pipeline/train_and_save.py
```

### 3. Generate Initial Batch Data
To populate the dashboard, run the batch prediction script over the sample `data/new_data.csv`.
*(Note: Ensure you set the `sys.path` properly or run with PYTHONPATH)*
```bash
$env:PYTHONPATH="."  # (Windows PowerShell)
export PYTHONPATH="." # (Linux/Mac)
python pipeline/batch_predict.py
```

### 4. Run FastAPI (Real-Time Service)
Serves the `/predict` endpoint for integrated services.
```bash
$env:PYTHONPATH="."
uvicorn api.app:app --reload --port 8000
```
- Access API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run Streamlit Dashboard
Displays all forecast operations and analytical charts.
```bash
$env:PYTHONPATH="."
streamlit run dashboard/streamlit_app.py
```
- Access Dashboard: [http://localhost:8501](http://localhost:8501)

---

## Docker Deployment (Optional)

To run the entire system via Docker:

```bash
docker build -t demand-forecast-app .
docker run -p 8000:8000 -p 8501:8501 demand-forecast-app
```
