import paho.mqtt.client as mqtt

class Device:
    def __init__(self, device_id, device_type):
        self.device_id = device_id
        self.device_type = device_type

    def send_data(self, data):
        # Send data to the IoT platform using MQTT
        pass

class IoTPlatform:
    def __init__(self):
        self.devices = {}

    def register_device(self, device):
        self.devices[device.device_id] = device

    def process_data(self, data):
        # Process and analyze the data
        pass

iot_platform = IoTPlatform()
device = Device('device1', 'temperature_sensor')
iot_platform.register_device(device)
device.send_data({'temperature': 25.5})
