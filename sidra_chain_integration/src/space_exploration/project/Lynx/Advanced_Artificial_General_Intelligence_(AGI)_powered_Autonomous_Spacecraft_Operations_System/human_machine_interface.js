import { HumanMachineInterface } from 'human-machine-interface-library';
import { SpacecraftModel } from 'spacecraft-model-library';

class HumanMachineInterface {
  constructor() {
    this.humanMachineInterface = new HumanMachineInterface();
    this.spacecraftModel = new SpacecraftModel();
  }

  update() {
    // Update human-machine interface with current spacecraft state
    this.humanMachineInterface.update(this.spacecraftModel.getSpacecraftState());
  }
}

export default HumanMachineInterface;
