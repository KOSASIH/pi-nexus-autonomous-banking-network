# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ..

CMD ["python", "did_based_autonomous_banking_api.py"]
