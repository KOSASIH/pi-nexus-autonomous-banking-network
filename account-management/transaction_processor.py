import asyncio
from kafka import KafkaProducer
from flink.core.execution import ExecutionEnvironment

class TransactionProcessor:
    def __init__(self, kafka_producer: KafkaProducer, execution_environment: ExecutionEnvironment):
        self.kafka_producer = kafka_producer
        self.execution_environment = execution_environment

    async def process_transaction(self, transaction: dict):
        # Produce transaction message to Kafka topic
        await self.kafka_producer.send("transactions", transaction)
        # Process transaction using Flink's DataStream API
        data_stream = self.execution_environment.add_source(transaction)
        data_stream.map(lambda x: x["amount"]).filter(lambda x: x > 1000).print()

if __name__ == "__main__":
    kafka_producer = KafkaProducer(bootstrap_servers=["localhost:9092"])
    execution_environment = ExecutionEnvironment.get_execution_environment()
    transaction_processor = TransactionProcessor(kafka_producer, execution_environment)
    asyncio.run(transaction_processor.process_transaction({"from_account": "Alice", "to_account": "Bob", "amount": 1500}))
