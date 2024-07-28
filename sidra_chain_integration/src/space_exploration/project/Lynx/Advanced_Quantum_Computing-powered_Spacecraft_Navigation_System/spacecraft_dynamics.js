import { SpacecraftModel } from 'spacecraft-model-library';

class SpacecraftDynamics {
  constructor() {
    this.spacecraftModel = new SpacecraftModel();
  }

  update(navigationData) {
    // Update spacecraft dynamics model with new navigation data
    this.spacecraftModel.update(navigationData);
  }

  getSpacecraftState() {
    // Return current spacecraft state (position, velocity, acceleration)
    return this.spacecraftModel.getSpacecraftState();
  }
}

export default SpacecraftDynamics;
