FROM nginx:alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY api-gateway.py /app/api-gateway.py
COPY config/ /app/config/
COPY middleware/ /app/middleware/
COPY static/ /app/static/

RUN pip install --no-cache-dir -r /app/config/requirements.txt

CMD ["python", "/app/api-gateway.py"]
