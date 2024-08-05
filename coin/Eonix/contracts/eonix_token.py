# eonix_token.py
import hashlib
import json

class EonixToken:
    def __init__(self, name, symbol, total_supply, decimals=18):
        self.name = name
        self.symbol = symbol
        self.total_supply = total_supply
        self.decimals = decimals
        self.token_id = self.generate_token_id()

    def generate_token_id(self):
        token_id_hash = hashlib.sha256(f"{self.name}{self.symbol}{self.total_supply}".encode()).hexdigest()
        return token_id_hash[:16]  # 16-character token ID

    def get_name(self):
        return self.name

    def get_symbol(self):
        return self.symbol

    def get_total_supply(self):
        return self.total_supply

    def get_decimals(self):
        return self.decimals

    def get_token_id(self):
        return self.token_id

    def to_dict(self):
        return {
            "name": self.name,
            "symbol": self.symbol,
            "total_supply": self.total_supply,
            "decimals": self.decimals,
            "token_id": self.token_id
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, token_dict):
        return cls(
            token_dict["name"],
            token_dict["symbol"],
            token_dict["total_supply"],
            token_dict["decimals"]
        )

    @classmethod
    def from_json(cls, token_json):
        token_dict = json.loads(token_json)
        return cls.from_dict(token_dict)
