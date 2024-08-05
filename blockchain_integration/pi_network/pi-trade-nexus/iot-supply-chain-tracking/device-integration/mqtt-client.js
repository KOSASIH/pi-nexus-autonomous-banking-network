// Import required libraries
const mqtt = require('mqtt');
const config = require('../config');

// Set up MQTT client
const client = mqtt.connect(config.mqttBrokerUrl, {
  clientId: config.clientId,
  username: config.username,
  password: config.password,
});

// Define MQTT topics
const topics = {
  telemetry: 'iot/supply-chain/tracking/telemetry',
  commands: 'iot/supply-chain/tracking/commands',
};

// Subscribe to MQTT topics
client.subscribe(topics.telemetry);
client.subscribe(topics.commands);

// Handle incoming MQTT messages
client.on('message', (topic, message) => {
  console.log(`Received message on topic ${topic}: ${message.toString()}`);

  // Handle telemetry data
  if (topic === topics.telemetry) {
    const data = JSON.parse(message.toString());
    console.log(`Received telemetry data: ${data}`);
    // Process telemetry data
  }

  // Handle commands
  if (topic === topics.commands) {
    const command = JSON.parse(message.toString());
    console.log(`Received command: ${command}`);
    // Process command
  }
});

// Publish MQTT messages
const publishTelemetryData = (data) => {
  client.publish(topics.telemetry, JSON.stringify(data));
};

const publishCommandResponse = (response) => {
  client.publish(topics.commands, JSON.stringify(response));
};

// Export MQTT client functions
module.exports = {
  publishTelemetryData,
  publishCommandResponse,
};
