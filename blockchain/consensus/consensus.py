from .pos import ProofOfStake
from .dpos import DelegatedProofOfStake

def create_consensus(consensus_type: str, validators: list, block_time: int = 10):
    if consensus_type == "pos":
        return ProofOfStake(validators, block_time)
    elif consensus_type == "dpos":
        return DelegatedProofOfStake(validators, block_time)
    else:
        raise ValueError("Invalid consensus type")

def run_consensus(consensus):
    consensus.run_consensus()
