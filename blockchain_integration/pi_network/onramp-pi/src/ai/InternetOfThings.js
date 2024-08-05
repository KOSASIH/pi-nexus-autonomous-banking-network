// InternetOfThings.js

import { Device } from 'device';

class InternetOfThings {
  constructor() {
    this.devices = [];
  }

    addDevice(device) {
    this.devices.push(device);
  }

  connectDevices() {
    // Connect all devices to the internet
    for (let i = 0; i < this.devices.length; i++) {
      const device = this.devices[i];
      device.connect();
    }
  }

  collectData() {
    // Collect data from all devices
    const data = [];
    for (let i = 0; i < this.devices.length; i++) {
      const device = this.devices[i];
      data.push(device.collectData());
    }
    return data;
  }
}

export default InternetOfThings;
