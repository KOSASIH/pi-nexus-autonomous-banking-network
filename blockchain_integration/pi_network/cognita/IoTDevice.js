import * as mqtt from 'qtt';

class IoTDevice {
  constructor() {
    this.client = mqtt.connect('mqtt://localhost:1883');
    this.client.on('connect', () => {
      console.log('Connected to MQTT broker');
    });
    this.client.on('message', (topic, message) => {
      console.log(`Received message on topic ${topic}: ${message}`);
    });
  }

  sendTelemetry(data) {
    this.client.publish('telemetry', JSON.stringify(data));
  }

  receiveCommand(command) {
    console.log(`Received command: ${command}`);
    // Handle command here
  }
}

export default IoTDevice;
