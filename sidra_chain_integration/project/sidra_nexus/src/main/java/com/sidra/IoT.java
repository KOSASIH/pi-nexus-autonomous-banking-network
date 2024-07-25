package com.sidra;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttMessage;

public class IoT {
    public static void main(String[] args) {
        // Set up an MQTT client
        MqttClient client = new MqttClient("tcp://localhost:1883", "client");

        // Publish a message
        MqttMessage message = new MqttMessage("Hello, world!".getBytes());
        client.publish("topic", message);
    }
}
