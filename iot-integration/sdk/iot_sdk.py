import json
import os
import socket
import threading

from cryptography.fernet import Fernet


class IoTSDK:

    def __init__(self, device_id, device_secret, sdk_secret):
        self.device_id = device_id
        self.device_secret = device_secret
        self.sdk_secret = sdk_secret
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(("localhost", 8080))
        self.cipher_suite = Fernet(self.sdk_secret)

    def send_data(self, data):
        encrypted_data = self.cipher_suite.encrypt(json.dumps(data).encode())
        self.socket.send(encrypted_data)

    def receive_command(self):
        encrypted_command = self.socket.recv(1024)
        command = self.cipher_suite.decrypt(encrypted_command).decode()
        return command

    def authenticate(self):
        encrypted_device_secret = self.cipher_suite.encrypt(self.device_secret.encode())
        self.socket.send(self.device_id.encode())
        self.socket.send(encrypted_device_secret)

    def start(self):
        self.authenticate()
        while True:
            command = self.receive_command()
            if command == "TRANSFER":
                amount = float(input("Enter amount to transfer: "))
                self.send_data({"type": "TRANSFER", "amount": amount})
            elif command == "BALANCE":
                self.send_data({"type": "BALANCE"})

    def stop(self):
        self.socket.close()


class SmartHome(IoTSDK):

    def __init__(self, device_id, device_secret, sdk_secret, home_id, home_secret):
        super().__init__(device_id, device_secret, sdk_secret)
        self.home_id = home_id
        self.home_secret = home_secret

    def send_home_data(self, data):
        self.send_data({"type": "HOME_DATA", "data": data})


class Wearable(IoTSDK):

    def __init__(self, device_id, device_secret, sdk_secret, user_id, user_secret):
        super().__init__(device_id, device_secret, sdk_secret)
        self.user_id = user_id
        self.user_secret = user_secret

    def send_health_data(self, data):
        self.send_data({"type": "HEALTH_DATA", "data": data})


class AutonomousVehicle(IoTSDK):

    def __init__(
        self, device_id, device_secret, sdk_secret, vehicle_id, vehicle_secret
    ):
        super().__init__(device_id, device_secret, sdk_secret)
        self.vehicle_id = vehicle_id
        self.vehicle_secret = vehicle_secret

    def send_gps_data(self, data):
        self.send_data({"type": "GPS_DATA", "data": data})

    def send_speed_data(self, data):
        self.send_data({"type": "SPEED_DATA", "data": data})


if __name__ == "__main__":
    device_id = "device123"
    device_secret = "secret123"
    sdk_secret = "sdk_secret123"
    home_id = "home123"
    home_secret = "secret123"
    user_id = "user123"
    user_secret = "secret123"
    vehicle_id = "vehicle123"
    vehicle_secret = "secret123"

    smart_home = SmartHome(device_id, device_secret, sdk_secret, home_id, home_secret)
    wearable = Wearable(device_id, device_secret, sdk_secret, user_id, user_secret)
    autonomous_vehicle = AutonomousVehicle(
        device_id, device_secret, sdk_secret, vehicle_id, vehicle_secret
    )

    smart_home.start()
    wearable.start()
    autonomous_vehicle.start()
