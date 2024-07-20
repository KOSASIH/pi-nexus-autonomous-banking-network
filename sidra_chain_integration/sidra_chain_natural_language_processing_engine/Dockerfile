# Dockerfile
FROM transformers/transformers:2.11.0

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "sidra_chain_natural_language_processing_engine.py"]
