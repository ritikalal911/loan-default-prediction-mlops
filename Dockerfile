FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app

COPY deployment_artifacts/model.joblib ./models/model.joblib
COPY deployment_artifacts/preprocessor.pkl ./data/processed/preprocessor.pkl

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]