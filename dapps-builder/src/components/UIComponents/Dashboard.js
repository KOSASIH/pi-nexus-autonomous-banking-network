import React from 'react';
import Button from './Button';

const Dashboard = ({ onTrain, onPredict }) => {
    return (
        <div>
            <h1>DApp Dashboard</h1>
            <Button onClick={onTrain} label="Train Model" />
            <Button onClick={onPredict} label="Make Prediction" />
        </div>
    );
};

export default Dashboard;
