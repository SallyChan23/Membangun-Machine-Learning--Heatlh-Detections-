FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY mlflow_model /opt/mlflow_model

RUN pip install mlflow

CMD ["mlflow", "models", "serve", "-m", "/opt/mlflow_model", "-h", "0.0.0.0", "-p", "5000"]
