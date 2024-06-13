import { ARJS } from 'ar.js';

class AugmentedReality {
  constructor() {
    this.arjs = new ARJS();
  }

  async renderARScene(scene) {
    const arScene = await this.arjs.renderScene(scene);
    return arScene;
  }
}

export default AugmentedReality;
