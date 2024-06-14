import React, { useState, useEffect } from 'eact';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const DataAnalytics = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get('/api/data-analytics')
     .then((response) => {
        setData(response.data);
      })
     .catch((error) => {
        console.error(error);
      });
  }, []);

  return (
    <div>
      <h2>Data Analytics</h2>
      <p>Visualize your financial data:</p>
      <LineChart width={500} height={300} data={data}>
        <Line type="monotone" dataKey="value" stroke="#8884d8" />
        <XAxis dataKey="date" />
        <YAxis />
        <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
        <Tooltip />
      </LineChart>
    </div>
  );
};

export default DataAnalytics;
