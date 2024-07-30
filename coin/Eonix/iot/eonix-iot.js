// Eonix IoT
const EonixIoT = {
  // Protocol
  protocol: 'MQTT',
  // Devices
  devices: [
    {
      name: 'EonixNode',
      type: 'Sensor',
      parameters: {
        temperature: 25,
        humidity: 60,
      },
    },
    {
      name: 'EonixSensor',
      type: 'Actuator',
      parameters: {
        led: 'on',
      },
    },
  ],
};
