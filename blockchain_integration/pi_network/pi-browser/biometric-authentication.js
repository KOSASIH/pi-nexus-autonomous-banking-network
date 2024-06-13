import { FaceAPI } from 'face-api.js';

class BiometricAuthentication {
  constructor() {
    this.faceApi = new FaceAPI();
  }

  async authenticateUser(faceImage) {
    const detection = await this.faceApi.detectFace(faceImage);
    const verification = await this.faceApi.verifyFace(detection);
    return verification;
  }
}

export default BiometricAuthentication;
