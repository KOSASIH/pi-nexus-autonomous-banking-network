import React, { useState, useEffect } from 'react';
import ChartComponent from './ChartComponent';
import TableComponent from './TableComponent';
import dataController from '../api/dataController';

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      const data = await dataController.getData();
      setData(data);
      setLoading(false);
    };
    fetchData();
  }, []);

  return (
    <div>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div>
          <ChartComponent data={data} />
          <TableComponent data={data} />
        </div>
      )}
    </div>
  );
};

export default Dashboard;
