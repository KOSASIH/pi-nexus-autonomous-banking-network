from .p2p import P2P
from .rpc import RPC

def start_p2p():
    p2p = P2P()
    asyncio.run(p2p.start())

def start_rpc():
    rpc = RPC()
    rpc.start()

def send_message(message: bytes):
    p2p = P2P()
    asyncio.run(p2p.send_message(message))

def call_rpc(service_name: str, method_name: str, request: grpc.Request) -> grpc.Response:
    rpc_client = RPCClient("localhost", 50051)
    return rpc_client.call(service_name, method_name, request)
