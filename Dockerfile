# Mode 2: Streamlit app container
FROM python:3.11-slim

WORKDIR /app

# System deps (optional)
RUN pip install --no-cache-dir --upgrade pip

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
# Dataset (needed to predict)
COPY FINAL_DATASET_READY_FOR_ML.csv ./FINAL_DATASET_READY_FOR_ML.csv

EXPOSE 8501

# Default: run Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
