class NetworkNode:
    def __init__(self, name):
        self.name = name
        self.data = 0

    def receive_data(self, amount):
        self.data += amount

class Router(NetworkNode):
    def __init__(self, name, connections):
        super().__init__(name)
        self.connections = connections

    def send_data(self, node, amount):
        node.receive_data(amount)

class Endpoint(NetworkNode):
    def __init__(self, name, ip_address):
        super().__init__(name)
        self.ip_address = ip_address
