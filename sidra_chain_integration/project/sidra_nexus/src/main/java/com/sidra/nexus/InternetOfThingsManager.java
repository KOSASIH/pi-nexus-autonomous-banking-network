package com.sidra.nexus;

import org.eclipse.paho.client.mqttv3.IMqttClient;
import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttException;

public class InternetOfThingsManager {
    private IMqttClient mqttClient;

    public InternetOfThingsManager() throws MqttException {
        mqttClient = new MqttClient("tcp://localhost:1883", "clientId");
        mqttClient.connect();
    }

    public void publishMessage(String topic, String message) throws MqttException {
        mqttClient.publish(topic, message.getBytes());
    }

    public void subscribeToTopic(String topic) throws MqttException {
        mqttClient.subscribe(topic);
    }
}
