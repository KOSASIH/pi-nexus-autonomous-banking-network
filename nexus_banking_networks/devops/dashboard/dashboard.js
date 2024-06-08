import React, { useState, useEffect } from 'eact';
import * as d3 from 'd3-array';
import { LineChart, Line, XAxis, YAxis } from 'eact-chartjs-2';

function Dashboard() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('/api/data')
     .then(response => response.json())
     .then(data => setData(data));
  }, []);

  const lineChart = (
    <LineChart data={data} options={{ title: { display: true, text: 'Transaction Volume' } }}>
      <Line type="line" dataKey="value" stroke="#8884d8" />
      <XAxis dataKey="date" />
      <YAxis />
    </LineChart>
  );

  return (
    <div>
      <h1>Autonomous Banking Network Dashboard</h1>
      {lineChart}
    </div>
  );
}

export default Dashboard;
