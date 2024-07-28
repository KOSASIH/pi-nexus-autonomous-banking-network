import { Spacecraft } from 'spacecraft-library';

class SpacecraftInterface {
  constructor(spacecraft) {
    this.spacecraft = spacecraft;
  }

  sendCommand(command) {
    // Send command to spacecraft
    this.spacecraft.sendCommand(command);
  }

  receiveData() {
    // Receive data from spacecraft
    const data = this.spacecraft.receiveData();
    return data;
  }
}

export default SpacecraftInterface;
