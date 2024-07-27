from typing import Dict

class SustainableEnergyTokenContract:
    def __init__(self):
        self.tokens: Dict[str, Dict[str, int]] = {}
        self.holders: Dict[str, Dict[str, int]] = {}

    def mint_token(self, token_id: str, amount: int, project_id: str):
        # Mint a new sustainable energy token
        self.tokens[token_id] = {
            "amount": amount,
            "project_id": project_id
        }
        self.holders[project_id] = self.holders.get(project_id, {})
        self.holders[project_id][token_id] = amount

    def transfer_token(self, from_holder: str, to_holder: str, token_id: str, amount: int):
        # Transfer a sustainable energy token from one holder to another
        if token_id not in self.tokens:
            raise ValueError("Token not found")
        if from_holder not in self.holders or token_id not in self.holders[from_holder]:
            raise ValueError("Holder not found")
        if self.holders[from_holder][token_id] < amount:
            raise ValueError("Insufficient balance")
        self.holders[from_holder][token_id] -= amount
        self.holders[to_holder] = self.holders.get(to_holder, {})
        self.holders[to_holder][token_id] = self.holders[to_holder].get(token_id, 0) + amount

    def burn_token(self, holder: str, token_id: str, amount: int):
        # Burn a sustainable energy token
        if token_id not in self.tokens:
            raise ValueError("Token not found")
        if holder not in self.holders or token_id not in self.holders[holder]:
            raise ValueError("Holder not found")
        if self.holders[holder][token_id] < amount:
            raise ValueError("Insufficient balance")
        self.holders[holder][token_id] -= amount
        if self.holders[holder][token_id] == 0:
            del self.holders[holder][token_id]

    def get_token_info(self, token_id: str):
        # Get the information of a sustainable energy token
        return self.tokens.get(token_id, {})

    def get_holder_balance(self, holder: str, token_id: str):
        # Get the balance of a holder for a token
        return self.holders.get(holder, {}).get(token_id, 0)

    def get_project_tokens(self, project_id: str):
        # Get the list of tokens associated with a project
        return [token_id for token_id, token_info in self.tokens.items() if token_info["project_id"] == project_id]
