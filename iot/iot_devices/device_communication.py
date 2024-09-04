import socket
import json

class DeviceCommunication:
    def __init__(self, device_ip: str, device_port: int):
        self.device_ip = device_ip
        self.device_port = device_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self) -> None:
        self.socket.connect((self.device_ip, self.device_port))

    def send_command(self, command: str) -> str:
        self.socket.sendall(command.encode())
        response = self.socket.recv(1024).decode()
        return response

    def send_data(self, data: dict) -> str:
        json_data = json.dumps(data)
        self.socket.sendall(json_data.encode())
        response = self.socket.recv(1024).decode()
        return response

    def close(self) -> None:
        self.socket.close()

# Example usage
comm = DeviceCommunication("192.168.1.100", 8080)
comm.connect()
response = comm.send_command("get_temperature")
print(response)  # Output: "25.5°C"
data = {"command": "set_humidity", "value": 60}
response = comm.send_data(data)
print(response)  # Output: "Humidity set to 60%"
comm.close()
