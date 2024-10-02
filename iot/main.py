import time

from iot.device import ATM, CreditCardTerminal, Sensor
from iot.mqtt import MQTT


class IoT:
    def __init__(self):
        self.atm = ATM("ATM 1", "atm/atm1")
        self.credit_card_terminal = CreditCardTerminal(
            "Credit Card Terminal 1", "credit_card/terminal1"
        )
        self.sensor = Sensor("Sensor 1", "sensor/sensor1")
        self.mqtt = MQTT("mqtt.example.com", 1883)

    def control_devices(self):
        while True:
            self.atm.withdraw_money(100)
            self.credit_card_terminal.read_card()
            data = self.sensor.read_sensor()
            message = f"Sensor 1: {data}"
            self.mqtt.publish(self.sensor.topic, message)
            time.sleep(1)

    def receive_messages(self):
        def on_message(client, userdata, message):
            topic = message.topic
            message = message.payload.decode()
            print(f"Received message: {message} from topic: {topic}")
            if topic == "atm/commands":
                command = message.split(":")[1].strip()
                self.atm.withdraw_money(int(command))
            elif topic == "credit_card/commands":
                command = message.split(":")[1].strip()
                self.credit_card_terminal.charge_card(int(command))

        self.mqtt.subscribe("atm/commands", on_message)
        self.mqtt.subscribe("credit_card/commands", on_message)
        self.mqtt.loop()


if __name__ == "__main__":
    iot = IoT()
    iot.control_devices()
    iot.receive_messages()
