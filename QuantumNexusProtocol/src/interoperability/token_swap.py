from web3 import Web3

class TokenSwap:
    def __init__(self, web3_instance, router_address):
        self.web3 = web3_instance
        self.router = self.web3.eth.contract(address=router_address, abi=router_abi)

    def swap_tokens(self, amount_in, amount_out_min, path, to, deadline):
        tx = self.router.functions.swapExactTokensForTokens(
            amount_in,
            amount_out_min,
            path,
            to,
            deadline
        ).transact()
        return self.web3.eth.waitForTransactionReceipt(tx)

# Example usage
if __name__ == "__main__":
    web3 = Web3(Web3.HTTPProvider('https://chain-url'))
    swap = TokenSwap(web3, '0xRouterAddress')
    receipt = swap.swap_tokens(amount_in=100, amount_out_min=90, path=['0xTokenA', '0xTokenB'], to='0xRecipient', deadline=1234567890)
    print(f"Swap Transaction Receipt: {receipt}")
