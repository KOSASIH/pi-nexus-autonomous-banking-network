from web3 import Web3

class CrossChainBridge:
    def __init__(self, source_chain_url, target_chain_url):
        self.source_web3 = Web3(Web3.HTTPProvider(source_chain_url))
        self.target_web3 = Web3(Web3.HTTPProvider(target_chain_url))

    def transfer_tokens(self, token_address, amount, target_address):
        # Approve the token transfer on the source chain
        token_contract = self.source_web3.eth.contract(address=token_address, abi=token_abi)
        tx = token_contract.functions.approve(self.target_web3.eth.defaultAccount, amount).transact()
        self.source_web3.eth.waitForTransactionReceipt(tx)

        # Transfer tokens to the bridge contract
        bridge_contract = self.source_web3.eth.contract(address=bridge_address, abi=bridge_abi)
        tx = bridge_contract.functions.lockTokens(token_address, amount, target_address).transact()
        self.source_web3.eth.waitForTransactionReceipt(tx)

        # Mint tokens on the target chain
        target_bridge_contract = self.target_web3.eth.contract(address=target_bridge_address, abi=target_bridge_abi)
        tx = target_bridge_contract.functions.mintTokens(target_address, amount).transact()
        self.target_web3.eth.waitForTransactionReceipt(tx)

# Example usage
if __name__ == "__main__":
    bridge = CrossChainBridge('https://source-chain-url', 'https://target-chain-url')
    bridge.transfer_tokens(token_address='0xTokenAddress', amount=100, target_address='0xTargetAddress')
