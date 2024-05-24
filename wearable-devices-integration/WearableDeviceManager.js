import { BleManager } from "react-native-ble-plx";

class WearableDeviceManager {
  constructor() {
    this.bleManager = new BleManager();
  }

  connectToDevice(device) {
    return this.bleManager.connectToDevice(device.id);
  }

  disconnectFromDevice(device) {
    return this.bleManager.disconnect(device.id);
  }

  readDataFromDevice(device, service, characteristic) {
    return this.bleManager.read(device.id, service, characteristic);
  }

  writeDataToDevice(device, service, characteristic, data) {
    return this.bleManager.write(device.id, service, characteristic, data);
  }

  startDataSync(device) {
    // Implement data synchronization logic here
  }

  stopDataSync(device) {
    // Implement data synchronization stop logic here
  }
}

export default new WearableDeviceManager();
