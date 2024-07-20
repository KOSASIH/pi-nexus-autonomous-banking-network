# Dockerfile
FROM tensorflow/tensorflow:2.4.0-gpu

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "sidra_chain_artificial_intelligence_engine.py"]
