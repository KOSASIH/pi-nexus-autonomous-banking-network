import { DigitalTwin } from 'digital-twin-sdk';

class DigitalTwin {
  constructor() {
    this.digitalTwin = new DigitalTwin();
  }

  async createDigitalTwin(asset) {
    const digitalTwin = await this.digitalTwin.create(asset);
    return digitalTwin;
  }

  async updateDigitalTwin(digitalTwin, updates) {
    const updatedDigitalTwin = await this.digitalTwin.update(digitalTwin, updates);
    return updatedDigitalTwin;
  }
}

export default DigitalTwin;
