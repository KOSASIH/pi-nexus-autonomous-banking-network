# app.py
from celery import Celery
from celery.utils.log import get_task_logger

app = Celery("transaction_processing", broker="amqp://guest:guest@localhost")

logger = get_task_logger(__name__)


@app.task
def process_transaction(transaction_id):
    # Process transaction
    logger.info(f"Processing transaction {transaction_id}")
    # Simulate processing time
    time.sleep(5)
    logger.info(f"Transaction {transaction_id} processed successfully")
