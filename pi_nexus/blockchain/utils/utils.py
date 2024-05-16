# blockchain/utils.py
from typing import Union

def calculate_hash(index: int, previous_hash: str, timestamp: datetime.datetime, data: Union[str, bytes]) -> str:
    """Calculate the hash of a block."""
    value = str(index) + str(previous_hash) + str(timestamp) + str(data)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()
