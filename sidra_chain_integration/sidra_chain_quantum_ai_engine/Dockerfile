# Dockerfile
FROM qiskit/qiskit-terra:0.21.0

WORKDIR /app

COPY requirements.txt.

RUN pip install -r requirements.txt

COPY..

CMD ["python", "sidra_chain_quantum_ai_engine.py"]
