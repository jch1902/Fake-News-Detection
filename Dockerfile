FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY src/ ./src/
COPY LSTMModel.keras .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

EXPOSE 5000


CMD ["gunicorn", "-b", "0.0.0.0:5000", "src.main:app"]