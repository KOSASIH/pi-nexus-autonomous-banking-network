import React, { useState, useEffect } from 'react';
import { FaceRecognition } from 'face-recognition-js';
import { FingerprintAnalysis } from 'fingerprint-analysis-js';
import { VoiceRecognition } from 'voice-recognition-js';

interface BiometricAuthenticationProps {
  user: any;
}

const BiometricAuthentication: React.FC<BiometricAuthenticationProps> = ({ user }) => {
  const [faceMatch, setFaceMatch] = useState(false);
  const [fingerprintMatch, setFingerprintMatch] = useState(false);
  const [voiceMatch, setVoiceMatch] = useState(false);

  useEffect(() => {
    const faceRecognition = new FaceRecognition();
    const fingerprintAnalysis = new FingerprintAnalysis();
    const voiceRecognition = new VoiceRecognition();

    faceRecognition.match(user.faceData).then((match) => {
      setFaceMatch(match);
    });

    fingerprintAnalysis.match(user.fingerprintData).then((match) => {
      setFingerprintMatch(match);
    });

    voiceRecognition.match(user.voiceData).then((match) => {
      setVoiceMatch(match);
    });
  }, [user]);

  return (
    <div>
      <h2>Biometric Authentication</h2>
      <p>Face Match: {faceMatch ? 'Yes' : 'No'}</p>
      <p>Fingerprint Match: {fingerprintMatch ? 'Yes' : 'No'}</p>
      <p>Voice Match: {voiceMatch ? 'Yes' : 'No'}</p>
    </div>
  );
};

export default BiometricAuthentication;
