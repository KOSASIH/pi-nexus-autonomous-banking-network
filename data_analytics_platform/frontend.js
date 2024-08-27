import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import axios from 'axios';

function App() {
  const [data, setData] = useState([]);
  const [aggregatedData, setAggregatedData] = useState([]);
  const [listingProgress, setListingProgress] = useState([]);

  useEffect(() => {
    axios.get('/api/data')
      .then(response => {
        setData(response.data);
      })
      .catch(error => {
        console.error(error);
      });

    axios.get('/api/aggregated_data')
      .then(response => {
        setAggregatedData(response.data);
      })
      .catch(error => {
        console.error(error);
      });

    axios.get('/api/listing_progress')
      .then(response => {
        setListingProgress(response.data);
      })
      .catch(error => {
        console.error(error);
      });
  }, []);

  return (
    <div>
      <h1>Pi Coin Real-Time Data Analytics</h1>
      <LineChart width={800} height={400} data={data}>
        <Line type="monotone" dataKey="price" stroke="#8884d8" />
        <XAxis dataKey="date" />
        <YAxis />
        <CartesianGrid stroke="#ccc" />
        <Tooltip />
      </LineChart>

      <h2>Aggregated Data</h2>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Mean Price</th>
            <th>Sum Volume</th>
          </tr>
        </thead>
        <tbody>
          {aggregatedData.map((row, index) => (
            <tr key={index}>
              <td>{row.date}</td>
              <td>{row.mean_price}</td>
              <td>{row.sum_volume}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Listing Progress</h2>
      <table>
        <thead>
          <tr>
            <th>Listing Status</th>
            <th>Count</th>
          </tr>
        </thead>
        <tbody>
          {listingProgress.map((row, index) => (
            <tr key={index}>
              <td>{row.listing_status}</td>
              <td>{row.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
