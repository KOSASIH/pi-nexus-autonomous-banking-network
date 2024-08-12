// NodePerformanceChart.js

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { useTheme } from 'styled-components';
import { useRecoilState } from 'recoil';
import { nodePerformanceDataState } from '../atoms/nodePerformanceData';

const NodePerformanceChart = () => {
  const theme = useTheme();
  const [nodePerformanceData, setNodePerformanceData] = useRecoilState(nodePerformanceDataState);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    const fetchNodePerformanceData = async () => {
      const response = await fetch('/api/node-performance-data');
      const data = await response.json();
      setNodePerformanceData(data);
    };
    fetchNodePerformanceData();
  }, []);

  useEffect(() => {
    const formatChartData = () => {
      const chartData = nodePerformanceData.map((node) => ({
        name: node.name,
        cpuUsage: node.performanceMetrics.cpuUsage,
        memoryUsage: node.performanceMetrics.memoryUsage,
      }));
      setChartData(chartData);
    };
    formatChartData();
  }, [nodePerformanceData]);

  return (
    <LineChart width={800} height={400} data={chartData}>
      <Line type="monotone" dataKey="cpuUsage" stroke={theme.colors.primary} />
      <Line type="monotone" dataKey="memoryUsage" stroke={theme.colors.secondary} />
      <XAxis dataKey="name" />
      <YAxis />
      <CartesianGrid stroke={theme.colors.grid} />
      <Tooltip />
    </LineChart>
  );
};

export default NodePerformanceChart;
