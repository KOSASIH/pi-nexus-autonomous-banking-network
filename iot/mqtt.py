import paho.mqtt.client as mqtt


class MQTT:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = mqtt.Client()

    def connect(self):
        self.client.connect(self.host, self.port)

    def publish(self, topic, message):
        self.client.publish(topic, message)

    def subscribe(self, topic, callback):
        self.client.subscribe(topic)
        self.client.on_message = callback

    def loop(self):
        self.client.loop_forever()
