from typing import Optional


class ConsensusRound:
    def __init__(self, round_number: int, validators: List[str], block_hash: str):
        self.round_number = round_number
        self.validators = validators
        self.block_hash = block_hash


class Consensus:
    def __init__(self, api_client):
        self.api_client = api_client

    def get_current_round(self) -> Optional[ConsensusRound]:
        # Implement Pi Network API call to get current consensus round
        pass

    def get_round_history(self) -> List[ConsensusRound]:
        # Implement Pi Network API call to get consensus round history
        pass
