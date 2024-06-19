class TransactionManager:
    def __init__(self, multi_chain_support):
        self.multi_chain_support = multi_chain_support

    async def send_transaction(self, chain_name, from_address, to_address, value):
        chain = self.multi_chain_support.get_chain(chain_name)
        if chain:
            return await chain.send_transaction(from_address, to_address, value)
        else:
            logger.error(f"Chain {chain_name} not supported")
            return None

    async def get_balance(self, chain_name, address):
        chain = self.multi_chain_support.get_chain(chain_name)
        if chain:
            return await chain.get_balance(address)
        else:
            logger.error(f"Chain {chain_name} not supported")
            return None
