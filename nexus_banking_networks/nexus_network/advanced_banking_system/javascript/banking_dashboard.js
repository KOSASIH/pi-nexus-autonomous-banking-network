// banking_dashboard.js
import React, { useState, useEffect } from 'eact';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'echarts';

function BankingDashboard() {
    const [data, setData] = useState([]);

    useEffect(() => {
        fetch('https://api.example.com/banking_data')
          .then(response => response.json())
          .then(data => setData(data));
    }, []);

    return (
        <div>
            <h1>Banking Dashboard</h1>
            <LineChart width={500} height={300} data={data}>
                <Line type="monotone" dataKey="balance" stroke="#8884d8" />
                <XAxis dataKey="date" />
                <YAxis />
                <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                <Tooltip />
            </LineChart>
        </div>
    );
}

export default BankingDashboard;
