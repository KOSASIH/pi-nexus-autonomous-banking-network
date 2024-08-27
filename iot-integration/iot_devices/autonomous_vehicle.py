import json
import os
import socket
import time

from cryptography.fernet import Fernet


class AutonomousVehicle:

    def __init__(self, vehicle_id, vehicle_secret, home_id, home_secret):
        self.vehicle_id = vehicle_id
        self.vehicle_secret = vehicle_secret
        self.home_id = home_id
        self.home_secret = home_secret
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(("localhost", 8080))
        self.gps_data = {"latitude": 37.7749, "longitude": -122.4194}
        self.speed_data = {"speed": 60}

    def send_gps_data(self):
        self.socket.send(json.dumps(self.gps_data).encode())

    def send_speed_data(self):
        self.socket.send(json.dumps(self.speed_data).encode())

    def receive_command(self):
        command = self.socket.recv(1024)
        return command.decode()

    def authenticate(self):
        cipher_suite = Fernet(self.vehicle_secret)
        encrypted_home_secret = cipher_suite.encrypt(self.home_secret.encode())
        self.socket.send(self.vehicle_id.encode())
        self.socket.send(encrypted_home_secret)

    def start(self):
        self.authenticate()
        while True:
            self.send_gps_data()
            self.send_speed_data()
            time.sleep(1)
            # Transfer $20.0 to bank account
            self.socket.send("TRANSFER:20.0".encode())
            time.sleep(1)
            self.socket.send("BALANCE".encode())  # Request current balance


if __name__ == "__main__":
    vehicle_id = "vehicle123"
    vehicle_secret = "secret123"
    home_id = "home123"
    home_secret = "secret123"
    autonomous_vehicle = AutonomousVehicle(
        vehicle_id, vehicle_secret, home_id, home_secret
    )
    autonomous_vehicle.start()
