import hashlib
import random

class LoadBalancer:
    def __init__(self, server_count):
        self.server_count = server_count
        self.servers = [None] * server_count

    def add_server(self, server):
        self.servers[server.id] = server

    def remove_server(self, server):
        self.servers[server.id] = None

    def route_request(self, request_id):
        hashed_id = hashlib.sha256(str(request_id).encode()).hexdigest()
        dest_server = int(hashed_id, 16) % self.server_count
        return self.servers[dest_server]
