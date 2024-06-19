import React, { useState, useEffect } from 'eact';
import { useWeb3React } from '@web3-react/core';
import { useContractLoader, useContractReader } from 'eth-hooks';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'echarts';

const SmartShipChart = () => {
  const { account, library } = useWeb3React();
  const contract = useContractLoader('SmartShip', library);
  const [transactions, setTransactions] = useState([]);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    async function fetchData() {
      const transactions = await contract.methods.getTransactions().call();
      setTransactions(transactions);
    }
    fetchData();
  }, [account, library]);

  useEffect(() => {
    const chartData = transactions.map((transaction) => ({
      timestamp: transaction.timestamp,
      amount: transaction.amount,
    }));
    setChartData(chartData);
  }, [transactions]);

  return (
    <LineChart width={500} height={300} data={chartData}>
      <Line type="monotone" dataKey="amount" stroke="#8884d8" />
      <XAxis dataKey="timestamp" />
      <YAxis />
      <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
      <Tooltip />
    </LineChart>
  );
};

export default SmartShipChart;
