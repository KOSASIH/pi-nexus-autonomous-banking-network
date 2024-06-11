import React, { useState, useEffect } from 'react';
import { TensorFlow } from 'tensorflow';
import { RiskManagementAPI } from '../api/risk-management';

interface AIpoweredRiskManagementProps {
  user: any;
}

const AIpoweredRiskManagement: React.FC<AIpoweredRiskManagementProps> = ({ user }) => {
  const [riskAssessment, setRiskAssessment] = useState({});

  useEffect(() => {
    const tensorflow = new TensorFlow();
    const riskManagementAPI = new RiskManagementAPI();

    tensorflow.loadModel('risk-management-model').then((model) => {
      const userInput = {
        transactions: user.transactions,
        creditScore: user.creditScore,
      };

      model.predict(userInput).then((output) => {
        setRiskAssessment(output);
      });
    });

    riskManagementAPI.getRiskAssessment(user.id).then((assessment) => {
      setRiskAssessment(assessment);
    });
  }, [user]);

  return (
    <div>
      <h2>AI-powered Risk Management</h2>
      <p>Risk Assessment: {JSON.stringify(riskAssessment)}</p>
    </div>
  );
};

export default AIpoweredRiskManagement;
