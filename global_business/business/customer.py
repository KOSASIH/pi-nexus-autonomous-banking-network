from dataclasses import dataclass
from uuid import uuid4


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    address: str
    phone_number: str

    def __post_init__(self):
        self.customer_id = str(uuid4())[:8]
