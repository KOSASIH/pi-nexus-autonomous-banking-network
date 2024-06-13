import { XR } from 'xr-sdk';

class ExtendedReality {
  constructor() {
    this.xr = new XR();
  }

  async renderXRExperience(experience) {
    const xrExperience = await this.xr.renderExperience(experience);
    return xrExperience;
  }
}

export default ExtendedReality;
