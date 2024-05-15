class Server:
    id = 0

    def __init__(self, load_balancer):
        self.load_balancer = load_balancer
        self.id = Server.id
        Server.id += 1

    def handle_request(self, request):
        print(f"Server {self.id} received request {request}")
        response = f"Response from Server {self.id} for request {request}"
        return response
