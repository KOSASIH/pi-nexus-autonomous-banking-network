import { TensorFlow } from 'tensorflow';

class ComputerVision {
  constructor() {
    this.tf = new TensorFlow();
  }

  async analyzeImage(image) {
    const analysis = await this.tf.analyzeImage(image);
    return analysis;
  }
}

export default ComputerVision;
