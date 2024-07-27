from typing import Dict

class CarbonCreditContract:
    def __init__(self):
        self.credit_owners: Dict[str, int] = {}
        self.credit_registry: Dict[str, Dict[str, str]] = {}

    def mint_credit(self, owner: str, amount: int, project_id: str):
        # Mint a new carbon credit
        self.credit_owners[owner] = self.credit_owners.get(owner, 0) + amount
        credit_id = f"CC-{len(self.credit_registry) + 1}"
        self.credit_registry[credit_id] = {
            "owner": owner,
            "amount": amount,
            "project_id": project_id
        }

    def transfer_credit(self, from_owner: str, to_owner: str, credit_id: str):
        # Transfer a carbon credit from one owner to another
        if credit_id not in self.credit_registry:
            raise ValueError("Credit not found")
        if self.credit_registry[credit_id]["owner"] != from_owner:
            raise ValueError("Credit owner mismatch")
        self.credit_registry[credit_id]["owner"] = to_owner
        self.credit_owners[from_owner] -= self.credit_registry[credit_id]["amount"]
        self.credit_owners[to_owner] = self.credit_owners.get(to_owner, 0) + self.credit_registry[credit_id]["amount"]

    def retire_credit(self, owner: str, credit_id: str):
        # Retire a carbon credit
        if credit_id not in self.credit_registry:
            raise ValueError("Credit not found")
        if self.credit_registry[credit_id]["owner"] != owner:
            raise ValueError("Credit owner mismatch")
        del self.credit_registry[credit_id]
        self.credit_owners[owner] -= self.credit_registry[credit_id]["amount"]

    def get_credit_balance(self, owner: str):
        # Get the carbon credit balance of an owner
        return self.credit_owners.get(owner, 0)

    def get_credit_info(self, credit_id: str):
        # Get the information of a carbon credit
        return self.credit_registry.get(credit_id, {})
