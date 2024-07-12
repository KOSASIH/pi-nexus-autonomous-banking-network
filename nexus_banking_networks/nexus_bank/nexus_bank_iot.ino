#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "my-wifi-ssid";
const char* password = "my-wifi-password";
const char* mqttServer = "my-mqtt-server";

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);
    while (WiFi.status()!= WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");
    client.setServer(mqttServer, 1883);
}

void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();
    delay(1000);
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("Attempting MQTT connection...");
        if (client.connect("NexusBankIoT")) {
            Serial.println("connected");
            client.publish("nexus_bank_iot", "Hello, World!");
        } else {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            Serial.println(" try again in 5 seconds");
            delay(5000);
        }
    }
}
