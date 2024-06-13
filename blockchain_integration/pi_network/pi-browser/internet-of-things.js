import { IoT } from 'iot-sdk';

class InternetOfThings {
  constructor() {
    this.iot = new IoT();
  }

  async connectToDevice(device) {
    const connection = await this.iot.connectToDevice(device);
    return connection;
  }

  async sendDataToDevice(device, data) {
    const sentData = await this.iot.sendDataToDevice(device, data);
    return sentData;
  }
}

export default InternetOfThings;
