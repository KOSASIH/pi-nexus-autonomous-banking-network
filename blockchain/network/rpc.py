import grpc
from concurrent import futures

from typing import Dict, List, Tuple

class RPC:
    def __init__(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self.services: Dict[str, grpc.Service] = {}

    def add_service(self, service: grpc.Service):
        self.services[service.name] = service
        self.server.add_generic_rpc_handlers((service,))

    def start(self):
        self.server.add_insecure_port('[::]:50051')
        self.server.start()

    def stop(self):
        self.server.stop(0)

class RPCService(grpc.Service):
    def __init__(self, name: str):
        self.name = name

    def handle_request(self, request: grpc.Request) -> grpc.Response:
        # Handle incoming requests
        pass

class RPCClient:
    def __init__(self, host: str, port: int):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = grpc.GenericStub(self.channel)

    def call(self, service_name: str, method_name: str, request: grpc.Request) -> grpc.Response:
        return self.stub.call(service_name, method_name, request)
