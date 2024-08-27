import json
import os
import socket
import threading

from cryptography.fernet import Fernet


class SmartHome:

    def __init__(self, home_id, home_secret, devices=None):
        self.home_id = home_id
        self.home_secret = home_secret
        self.devices = devices if devices else []
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("localhost", 8080))
        self.socket.listen(5)
        self.threads = []

    def add_device(self, device):
        self.devices.append(device)

    def remove_device(self, device):
        self.devices.remove(device)

    def send_command(self, device_id, command):
        for device in self.devices:
            if device["id"] == device_id:
                device["socket"].send(command.encode())
                break

    def receive_data(self, device_id):
        for device in self.devices:
            if device["id"] == device_id:
                data = device["socket"].recv(1024)
                return data.decode()

    def start_listening(self):
        while True:
            client_socket, address = self.socket.accept()
            device_id = client_socket.recv(1024).decode()
            device_secret = client_socket.recv(1024).decode()
            if self.authenticate_device(device_id, device_secret):
                self.devices.append({"id": device_id, "socket": client_socket})
                thread = threading.Thread(
                    target=self.handle_device, args=(client_socket,)
                )
                thread.start()
                self.threads.append(thread)

    def handle_device(self, client_socket):
        while True:
            data = client_socket.recv(1024)
            if data:
                print(
                    f"Received data from device {client_socket.getpeername()}: {data.decode()}"
                )
            else:
                break

    def authenticate_device(self, device_id, device_secret):
        cipher_suite = Fernet(self.home_secret)
        encrypted_device_secret = cipher_suite.encrypt(device_secret.encode())
        return encrypted_device_secret == device_secret.encode()

    def start(self):
        thread = threading.Thread(target=self.start_listening)
        thread.start()
        self.threads.append(thread)

    def stop(self):
        for thread in self.threads:
            thread.join()


if __name__ == "__main__":
    home_id = "home123"
    home_secret = "secret123"
    smart_home = SmartHome(home_id, home_secret)
    smart_home.start()
