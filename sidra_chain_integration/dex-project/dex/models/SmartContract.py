class SmartContract:
    def __init__(self, address, bytecode):
        self.address = address
        self.bytecode = bytecode
        self.storage = {}

    def deploy(self, blockchain):
        blockchain.add_transaction({"from": "deployer", "to": self.address, "amount": 0})

    def call(self, blockchain, function, args):
        # execute the function on the contract's bytecode
        # update the contract's storage
        pass

    def get_storage(self):
        return self.storage
