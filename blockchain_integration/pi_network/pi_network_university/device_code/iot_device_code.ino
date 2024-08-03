// iot_device_code.ino

#include <WiFi.h>
#include <PubSubClient.h>

// WiFi credentials
const char* ssid = "your_wifi_ssid";
const char* password = "your_wifi_password";

// MQTT broker credentials
const char* mqttServer = "your_mqtt_broker_url";
const char* mqttTopic = "your_mqtt_topic";
const char* mqttUsername = "your_mqtt_username";
const char* mqttPassword = "your_mqtt_password";

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  Serial.begin(115200);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.println("Initializing MQTT client...");

  // Connect to MQTT broker
  client.setServer(mqttServer, 1883);
  client.connect(mqttUsername, mqttPassword);
}

void loop() {
  // Read sensor data (e.g. temperature, humidity, etc.)
  int sensorValue = analogRead(A0);

  // Convert sensor data to string
  String sensorData = String(sensorValue);

  // Publish sensor data to MQTT topic
  client.publish(mqttTopic, sensorData.c_str());

  // Wait for 1 minute before publishing again
  delay(60000);
}
