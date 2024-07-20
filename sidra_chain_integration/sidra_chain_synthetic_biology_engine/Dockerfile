# Dockerfile
FROM synbio/synbio:1.0.0

WORKDIR /app

COPY requirements.txt.

RUN pip install -r requirements.txt

COPY..

CMD ["python", "sidra_chain_synthetic_biology_engine.py"]
