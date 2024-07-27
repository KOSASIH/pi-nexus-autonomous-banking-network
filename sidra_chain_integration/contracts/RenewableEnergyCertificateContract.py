from typing import Dict

class RenewableEnergyCertificateContract:
    def __init__(self):
        self.rec_owners: Dict[str, int] = {}
        self.rec_registry: Dict[str, str] = {}

    def mint_rec(self, owner: str, amount: int, renewable_energy_source: str):
        # Mint a new REC
        self.rec_owners[owner] = self.rec_owners.get(owner, 0) + amount
        rec_id = f"REC-{len(self.rec_registry) + 1}"
        self.rec_registry[rec_id] = {
            "owner": owner,
            "amount": amount,
            "renewable_energy_source": renewable_energy_source
        }

    def transfer_rec(self, from_owner: str, to_owner: str, rec_id: str):
        # Transfer a REC from one owner to another
        if rec_id not in self.rec_registry:
            raise ValueError("REC not found")
        if self.rec_registry[rec_id]["owner"] != from_owner:
            raise ValueError("REC owner mismatch")
        self.rec_registry[rec_id]["owner"] = to_owner
        self.rec_owners[from_owner] -= self.rec_registry[rec_id]["amount"]
        self.rec_owners[to_owner] = self.rec_owners.get(to_owner, 0) + self.rec_registry[rec_id]["amount"]

    def retire_rec(self, owner: str, rec_id: str):
        # Retire a REC
        if rec_id not in self.rec_registry:
            raise ValueError("REC not found")
        if self.rec_registry[rec_id]["owner"] != owner:
            raise ValueError("REC owner mismatch")
        del self.rec_registry[rec_id]
        self.rec_owners[owner] -= self.rec_registry[rec_id]["amount"]

    def get_rec_balance(self, owner: str):
        # Get the REC balance of an owner
        return self.rec_owners.get(owner, 0)

    def get_rec_info(self, rec_id: str):
        # Get the information of a REC
        return self.rec_registry.get(rec_id, {})
