# pi_nft.py

import web3
from web3.contract import Contract

class PINFT:
    def __init__(self, web3: web3.Web3, contract_address: str):
        self.web3 = web3
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.get_abi())

    def get_abi(self) -> list:
        # Load the PI NFT ABI from a file or database
        with open('pi_nft.abi', 'r') as f:
            return json.load(f)

    def create_nft(self, metadata: dict) -> int:
        # Create a new NFT
        tx_hash = self.contract.functions.createNFT(metadata).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).logs[0].args.tokenId

    def get_nft_metadata(self, token_id: int) -> dict:
        # Get the metadata for an NFT
        return self.contract.functions.getNFTMetadata(token_id).call()

    def transfer_nft(self, to: str, token_id: int) -> bool:
        # Transfer an NFT to another user
        tx_hash = self.contract.functions.transferNFT(to, token_id).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1

    def set_nft_parameters(self, parameters: dict) -> bool:
        # Set the parameters for the NFT contract
        tx_hash = self.contract.functions.setNFTParameters(parameters).transact({'from': self.web3.eth.accounts[0]})
        return self.web3.eth.waitForTransactionReceipt(tx_hash).status == 1
