import { HCI } from 'hci-sdk';

class HumanComputerInteraction {
  constructor() {
    this.hci = new HCI();
  }

  async analyzeUserBehavior(behavior) {
    const insights = await this.hci.analyze(behavior);
    return insights;
  }
}

export default HumanComputerInteraction;
