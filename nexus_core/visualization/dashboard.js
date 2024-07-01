import React, { useState, useEffect } from 'eact';
import { Chart } from 'chart.js';
import { Grid, Typography, Button } from '@material-ui/core';
import axios from 'axios';

const Dashboard = () => {
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/api/data');
        setData(response.data);
        setLoading(false);
      } catch (error) {
        setError(error.message);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const handleRefresh = () => {
    setLoading(true);
    fetchData();
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h4">Data Visualization Dashboard</Typography>
      </Grid>
      <Grid item xs={12}>
        {loading? (
          <Typography>Loading...</Typography>
        ) : (
          <Chart type="line" data={data} />
        )}
      </Grid>
      <Grid item xs={12}>
        {error && <Typography color="error">{error}</Typography>}
      </Grid>
      <Grid item xs={12}>
        <Button variant="contained" color="primary" onClick={handleRefresh}>
          Refresh Data
        </Button>
      </Grid>
    </Grid>
  );
};

export default Dashboard;
