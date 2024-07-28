import { SpacecraftModel } from 'spacecraft-model-library';

class SpacecraftControl {
  constructor() {
    this.spacecraftModel = new SpacecraftModel();
  }

  execute(decision) {
    // Execute decision on spacecraft systems
    this.spacecraftModel.execute(decision);
  }
}

export default SpacecraftControl;
