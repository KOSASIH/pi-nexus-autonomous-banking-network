import socket


class Networking:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8080
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))

    def listen_for_connections(self):
        self.socket.listen(5)
        while True:
            connection, address = self.socket.accept()
            print(f"Connection from {address} has been established!")
            self.handle_connection(connection)

    def handle_connection(self, connection):
        while True:
            data = connection.recv(1024)
            if not data:
                break
            connection.send(data)
        connection.close()
