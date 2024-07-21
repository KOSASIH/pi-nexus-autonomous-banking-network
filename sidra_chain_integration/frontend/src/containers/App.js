import React from 'react';
import LoanApplication from '../components/LoanApplication';
import InvestmentDashboard from '../components/InvestmentDashboard';
import RiskAssessment from '../components/RiskAssessment';

const App = () => {
    return (
        <div>
            <h1>Autonomous Banking System</h1>
            <LoanApplication />
            <InvestmentDashboard />
            <RiskAssessment />
        </div>
    );
};

export default App;
