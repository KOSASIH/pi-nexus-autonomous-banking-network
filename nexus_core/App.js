import React, { useState, useEffect } from 'react';
import ChartComponent from './components/ChartComponent';
import TableComponent from './components/TableComponent';
import getData from './api/data';
import formatData from './utils/formatData';

const App = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      const data = await getData();
      setData(formatData(data));
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

export default App;
