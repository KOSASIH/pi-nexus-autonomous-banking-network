import { CyberPhysicalSystem } from 'cyber-physical-system-sdk';

class CyberPhysicalSystems {
  constructor() {
    this.cyberPhysicalSystem = new CyberPhysicalSystem();
  }

  async integratePhysicalSystem(system) {
    const integratedSystem = await this.cyberPhysicalSystem.integrate(system);
    return integratedSystem;
  }
}

export default CyberPhysicalSystems;
