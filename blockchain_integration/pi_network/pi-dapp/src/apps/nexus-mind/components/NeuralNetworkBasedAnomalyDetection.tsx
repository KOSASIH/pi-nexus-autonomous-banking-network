import React, { useState, useEffect } from 'react';
import { BrainJS } from 'brain.js';
import { AnomalyDetectionAPI } from '../api/anomaly-detection';

interface NeuralNetworkBasedAnomalyDetectionProps {
  user: any;
}

const NeuralNetworkBasedAnomalyDetection: React.FC<NeuralNetworkBasedAnomalyDetectionProps> = ({ user }) => {
  const [anomalyDetected, setAnomalyDetected] = useState(false);

  useEffect(() => {
    const brainJS = new BrainJS();
    const anomalyDetectionAPI = new AnomalyDetectionAPI();

    brainJS.train(user.transactionData).then((network) => {
      const output = network.run(user.transactionData);
      if (output > 0.5) {
        setAnomalyDetected(true);
      }
    });

    anomalyDetectionAPI.detectAnomaly(user.id).then((detected) => {
      setAnomalyDetected(detected);
    });
  }, [user]);

  return (
    <div>
      <h2>Neural Network-based Anomaly Detection</h2>
      <p>Anomaly Detected: {anomalyDetected ? 'Yes' : 'No'}</p>
    </div>
  );
};

export default NeuralNetworkBasedAnomalyDetection;
