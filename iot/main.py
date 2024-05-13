import time
from iot.device import Device
from iot.mqtt import MQTT

class IoT:
    def __init__(self):
        self.devices = [
            Device("Device 1", "devices/device1"),
            Device("Device 2", "devices/device2"),
            Device("Device 3", "devices/device3")
        ]
        self.mqtt = MQTT("mqtt.example.com", 1883)

    def control_devices(self):
        for device in self.devices:
            data = device.read_sensor()
            message = f"{device.name}: {data}"
            self.mqtt.publish(device.topic, message)
            time.sleep(1)

    def receive_messages(self):
        def on_message(client, userdata, message):
            topic = message.topic
           message = message.payload.decode()
            print(f"Received message: {message} from topic: {topic}")

        self.mqtt.subscribe("commands/#", on_message)
        self.mqtt.loop()

if __name__ == "__main__":
    iot = IoT()
    iot.control_devices()
    iot.receive_messages()
