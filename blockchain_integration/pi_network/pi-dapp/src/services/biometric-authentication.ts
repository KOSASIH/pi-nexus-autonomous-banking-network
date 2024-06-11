import { FaceRecognition } from 'face-recognition-js';
import { FingerprintAnalysis } from 'fingerprint-analysis-js';
import { VoiceRecognition } from 'voice-recognition-js';

class BiometricAuthentication {
  private faceRecognition: FaceRecognition;
  private fingerprintAnalysis: FingerprintAnalysis;
  private voiceRecognition: VoiceRecognition;

  constructor() {
    this.faceRecognition = new FaceRecognition();
    this.fingerprintAnalysis = new FingerprintAnalysis();
    this.voiceRecognition = new VoiceRecognition();
  }

  async authenticateUser(user: any): Promise<boolean> {
    const faceMatch = await this.faceRecognition.match(user.faceData);
    const fingerprintMatch = await this.fingerprintAnalysis.match(user.fingerprintData);
    const voiceMatch = await this.voiceRecognition.match(user.voiceData);
    return faceMatch && fingerprintMatch && voiceMatch;
  }
}

export default BiometricAuthentication;
