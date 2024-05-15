import time


class Client:
    def __init__(self, load_balancer):
        self.load_balancer = load_balancer

    def send_request(self, request_id):
        server = self.load_balancer.route_request(request_id)
        response = server.handle_request(request_id)
        print(f"Client received response {response}")


if __name__ == "__main__":
    load_balancer = LoadBalancer(10)

    server1 = Server(load_balancer)
    server2 = Server(load_balancer)
    server3 = Server(load_balancer)

    load_balancer.add_server(server1)
    load_balancer.add_server(server2)
    load_balancer.add_server(server3)

    client = Client(load_balancer)

    for i in range(10):
        client.send_request(i)
        time.sleep(1)
