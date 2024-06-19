class WalletManager:
    def __init__(self, multi_chain_support):
        self.multi_chain_support = multi_chain_support
        self.wallets = {}

    async def create_wallet(self, chain_name):
        chain = self.multi_chain_support.get_chain(chain_name)
        if chain:
            wallet = await chain.create_wallet()
            self.wallets[chain_name] = wallet
            return wallet
        else:
            logger.error(f"Chain {chain_name} not supported")
            return None

    async def get_wallet(self, chain_name):
        return self.wallets.get(chain_name)

    async def delete_wallet(self, chain_name):
        if chain_name in self.wallets:
            del self.wallets[chain_name]
