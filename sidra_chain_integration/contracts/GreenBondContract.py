from typing import Dict

class GreenBondContract:
    def __init__(self):
        self.bonds: Dict[str, Dict[str, str]] = {}
        self.issuers: Dict[str, Dict[str, str]] = {}

    def issue_bond(self, bond_id: str, issuer_id: str, amount: int, project_id: str):
        # Issue a new green bond
        self.bonds[bond_id] = {
            "issuer_id": issuer_id,
            "amount": amount,
            "project_id": project_id,
            "holders": []
        }
        self.issuers[issuer_id] = self.issuers.get(issuer_id, {})
        self.issuers[issuer_id][bond_id] = bond_id

    def transfer_bond(self, from_holder: str, to_holder: str, bond_id: str):
        # Transfer a green bond from one holder to another
        if bond_id not in self.bonds:
            raise ValueError("Bond not found")
        if from_holder not in self.bonds[bond_id]["holders"]:
            raise ValueError("Holder not found")
        self.bonds[bond_id]["holders"].remove(from_holder)
        self.bonds[bond_id]["holders"].append(to_holder)

    def redeem_bond(self, holder: str, bond_id: str):
        # Redeem a green bond
        if bond_id not in self.bonds:
            raise ValueError("Bond not found")
        if holder not in self.bonds[bond_id]["holders"]:
            raise ValueError("Holder not found")
        self.bonds[bond_id]["holders"].remove(holder)
        del self.bonds[bond_id]

    def get_bond_info(self, bond_id: str):
        # Get the information of a green bond
        return self.bonds.get(bond_id, {})

    def get_issuer_bonds(self, issuer_id: str):
        # Get the list of bonds issued by an issuer
        return list(self.issuers.get(issuer_id, {}).values())
