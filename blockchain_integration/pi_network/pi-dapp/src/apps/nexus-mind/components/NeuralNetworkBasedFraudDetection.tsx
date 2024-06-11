import React, { useState, useEffect } from 'react';
import { BrainJS } from 'brain.js';
import { FraudDetectionAPI } from '../api/fraud-detection';

interface NeuralNetworkBasedFraudDetectionProps {
  user: any;
}

const NeuralNetworkBasedFraudDetection: React.FC<NeuralNetworkBasedFraudDetectionProps> = ({ user }) => {
  const [fraudDetected, setFraudDetected] = useState(false);

  useEffect(() => {
    const brainJS = new BrainJS();
    const fraudDetectionAPI = new FraudDetectionAPI();

    brainJS.train(user.transactionData).then((network) => {
      const output = network.run(user.transactionData);
      if (output > 0.5) {
        setFraudDetected(true);
      }
    });

    fraudDetectionAPI.detectFraud(user.id).then((detected) => {
      setFraudDetected(detected);
    });
  }, [user]);

  return (
    <div>
      <h2>Neural Network-based Fraud Detection</h2>
      <p>Fraud Detected: {fraudDetected ? 'Yes' : 'No'}</p>
    </div>
  );
};

export default NeuralNetworkBasedFraudDetection;
