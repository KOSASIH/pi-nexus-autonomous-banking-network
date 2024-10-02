# transaction_monitor.py

import json
import logging
from typing import Dict, List

import requests


class TransactionMonitor:
    def __init__(
        self, pi_network_api_key: str, pi_network_api_base_url: str, webhook_url: str
    ):
        self.pi_network_api_key = pi_network_api_key
        self.pi_network_api_base_url = pi_network_api_base_url
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)

    def get_transactions(self) -> List[Dict]:
        headers = {
            "Authorization": f"Bearer {self.pi_network_api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"{self.pi_network_api_base_url}/transactions", headers=headers
        )
        response.raise_for_status()
        return response.json()

    def monitor_transactions(self) -> None:
        while True:
            transactions = self.get_transactions()
            for transaction in transactions:
                self.process_transaction(transaction)

    def process_transaction(self, transaction: Dict) -> None:
        transaction_id = transaction["id"]
        transaction_status = transaction["status"]
        if transaction_status == "pending":
            self.logger.info(f"Transaction {transaction_id} is pending")
            self.send_webhook_notification(transaction_id, "pending")
        elif transaction_status == "confirmed":
            self.logger.info(f"Transaction {transaction_id} is confirmed")
            self.send_webhook_notification(transaction_id, "confirmed")
        elif transaction_status == "failed":
            self.logger.error(f"Transaction {transaction_id} failed")
            self.send_webhook_notification(transaction_id, "failed")

    def send_webhook_notification(self, transaction_id: str, status: str) -> None:
        payload = {"transaction_id": transaction_id, "status": status}
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.webhook_url, headers=headers, json=payload)
        response.raise_for_status()


if __name__ == "__main__":
    config = Config()
    pi_network_api_key = config.get_pi_network_api_key()
    pi_network_api_base_url = config.get_pi_network_api_base_url()
    webhook_url = "https://example.com/webhook"
    transaction_monitor = TransactionMonitor(
        pi_network_api_key, pi_network_api_base_url, webhook_url
    )
    transaction_monitor.monitor_transactions()
