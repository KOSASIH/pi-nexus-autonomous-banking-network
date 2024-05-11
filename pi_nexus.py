# Original code
def send_message(chat_id, message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, json=data)

# Improved code
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def send_message(chat_id: int, message: str) -> None:
    """Send a message to a Telegram chat"""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data: Dict[str, str] = {"chat_id": str(chat_id), "text": message}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error sending message: {e}")
