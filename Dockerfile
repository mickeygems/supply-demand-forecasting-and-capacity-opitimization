FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Ensure PYTHONPATH is set so local imports work
ENV PYTHONPATH=/app

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# The default command will run both FastAPI and Streamlit using a simple shell script
# Note: For production, it is typically recommended to separate these into two containers.
CMD uvicorn api.app:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
