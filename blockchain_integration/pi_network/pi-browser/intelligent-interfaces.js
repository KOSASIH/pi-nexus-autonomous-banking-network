import { IntelligentInterface } from 'intelligent-interface-sdk';

class IntelligentInterfaces {
  constructor() {
    this.intelligentInterface = new IntelligentInterface();
  }

  async createIntelligentInterface(interfaceConfig) {
    const interface = await this.intelligentInterface.create(interfaceConfig);
    return interface;
  }
}

export default IntelligentInterfaces;
