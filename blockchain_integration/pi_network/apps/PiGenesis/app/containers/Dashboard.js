import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { useWeb3React } from '@web3-react/core';
import { Web3Provider } from '@ethersproject/providers';
import { useEthers } from '@ethersproject/ethers-react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBitcoin, faEthereum } from '@fortawesome/free-brands-svg-icons';

const Dashboard = () => {
  const [chartData, setChartData] = useState([]);
  const [accountBalance, setAccountBalance] = useState(0);
  const [piTokenBalance, setPiTokenBalance] = useState(0);

  const { account, library } = useWeb3React();
  const { ethers } = useEthers();

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const response = await fetch('https://api.pi-genesis.com/chart-data');
        const data = await response.json();
        setChartData(data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchChartData();
  }, []);

  useEffect(() => {
    const fetchAccountBalance = async () => {
      try {
        const response = await library.send('eth_getBalance', [account, 'latest']);
        const balance = await response.json();
        setAccountBalance(balance);
      } catch (error) {
        console.error(error);
      }
    };
    fetchAccountBalance();
  }, [account, library]);

  useEffect(() => {
    const fetchPiTokenBalance = async () => {
      try {
        const response = await library.send('eth_call', [
          {
            to: '0x...PiTokenContractAddress...',
            data: '0x...getBalance...',
          },
          'latest',
        ]);
        const balance = await response.json();
        setPiTokenBalance(balance);
      } catch (error) {
        console.error(error);
      }
    };
    fetchPiTokenBalance();
  }, [account, library]);

  return (
    <div className="dashboard">
      <h1>Dashboard</h1>
      <div className="row">
        <div className="col-md-6">
          <h2>Account Balance</h2>
          <p>
            <FontAwesomeIcon icon={faEthereum} size="lg" />
            {accountBalance} ETH
          </p>
        </div>
        <div className="col-md-6">
          <h2>Pi Token Balance</h2>
          <p>
            <FontAwesomeIcon icon={faBitcoin} size="lg" />
            {piTokenBalance} PGT
          </p>
        </div>
      </div>
      <div className="row">
        <div className="col-md-12">
          <h2>Market Chart</h2>
          <LineChart width={800} height={400} data={chartData}>
            <Line type="monotone" dataKey="price" stroke="#8884d8" />
            <XAxis dataKey="date" />
            <YAxis />
            <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
            <Tooltip />
          </LineChart>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
