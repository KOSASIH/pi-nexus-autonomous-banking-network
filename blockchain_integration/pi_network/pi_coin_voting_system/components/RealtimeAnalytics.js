import React, { useState, useEffect } from 'eact';
import { connect } from 'eact-redux';
import { getAnalytics } from '../actions/analytics.actions';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'echarts';
import { WebSocket } from 'ws';

const RealtimeAnalytics = ({ getAnalytics, analyticsData }) => {
  const [ws, setWs] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const wsUrl = 'wss://example.com/analytics-ws';
    const wsOptions = {
      headers: {
        Authorization: 'Bearer YOUR_TOKEN',
      },
    };
    const ws = new WebSocket(wsUrl, wsOptions);
    setWs(ws);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setChartData((prevData) => [...prevData, data]);
    };

    ws.onopen = () => {
      console.log('Connected to WebSocket');
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    if (analyticsData) {
      const data = analyticsData.map((item) => ({
        time: item.timestamp,
        value: item.value,
      }));
      setChartData(data);
      setLoading(false);
    } else {
      setLoading(true);
    }
  }, [analyticsData]);

  return (
    <div>
      <h1>Realtime Analytics</h1>
      {loading ? (
        <Spinner color="primary" />
      ) : (
        <LineChart width={800} height={400} data={chartData}>
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
          <XAxis dataKey="time" />
          <YAxis />
          <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
          <Tooltip />
        </LineChart>
      )}
    </div>
  );
};

const mapStateToProps = (state) => {
  return {
    analyticsData: state.analytics.data,
  };
};

export default connect(mapStateToProps, { getAnalytics })(RealtimeAnalytics);
