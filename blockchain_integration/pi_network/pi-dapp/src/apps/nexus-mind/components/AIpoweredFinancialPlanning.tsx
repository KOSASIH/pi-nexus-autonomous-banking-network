import React, { useState, useEffect } from 'react';
import { TensorFlow } from 'tensorflow';
import { FinancialPlanningAPI } from '../api/financial-planning';

interface AIpoweredFinancialPlanningProps {
  user: any;
}

const AIpoweredFinancialPlanning: React.FC<AIpoweredFinancialPlanningProps> = ({ user }) => {
  const [financialPlan, setFinancialPlan] = useState({});

  useEffect(() => {
    const tensorflow = new TensorFlow();
    const financialPlanningAPI = new FinancialPlanningAPI();

    tensorflow.loadModel('financial-planning-model').then((model) => {
      const userInput = {
        income: user.income,
        expenses: user.expenses,
        goals: user.goals,
      };

      model.predict(userInput).then((output) => {
        setFinancialPlan(output);
      });
    });

    financialPlanningAPI.getFinancialPlan(user.id).then((plan) => {
      setFinancialPlan(plan);
    });
  }, [user]);

  return (
    <div>
      <h2>AI-powered Financial Planning</h2>
      <p>Financial Plan: {JSON.stringify(financialPlan)}</p>
    </div>
  );
};

export default AIpoweredFinancialPlanning;
