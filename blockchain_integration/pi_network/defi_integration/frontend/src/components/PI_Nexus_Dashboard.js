import React, { useState, useEffect, useContext, useRef } from 'react';
import Web3 from 'web3';
import { useWeb3React } from '@web3-react/core';
import { Contract } from 'web3/eth/contract';
import PI_Nexus_ABI from '../../../contracts/PI_Nexus_Autonomous_Banking_Network_v3.json';
import { ThemeContext } from '../../contexts/ThemeContext';
import { WalletContext } from '../../contexts/WalletContext';
import { NotificationContext } from '../../contexts/NotificationContext';
import './PI_Nexus_Dashboard.css';
import { Chart } from 'react-chartjs-2';
import { LineChart, Line, CartesianGrid, XAxis, YAxis } from 'recharts';
import { useMediaQuery } from 'react-responsive';
import { useInterval } from 'react-use';
import { debounce } from 'lodash';
import { toast } from 'react-toastify';

const PI_NEXUS_CONTRACT_ADDRESS = '0x...'; // Replace with actual contract address

const PI_Nexus_Dashboard = () => {
  const { active, account, library } = useWeb3React();
  const { theme } = useContext(ThemeContext);
  const { wallet } = useContext(WalletContext);
  const { notify } = useContext(NotificationContext);
  const [piNexusContract, setPiNexusContract] = useState(null);
  const [governanceProposals, setGovernanceProposals] = useState([]);
  const [rewards, setRewards] = useState([]);
  const [liquidity, setLiquidity] = useState(0);
  const [borrowingRequests, setBorrowingRequests] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [isMobile] = useMediaQuery({ query: '(max-width: 768px)' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (active && library && !piNexusContract) {
      const web3 = new Web3(library.provider);
      const contract = new Contract(
        PI_NEXUS_CONTRACT_ADDRESS,
        PI_Nexus_ABI,
        web3.currentProvider
      );
      setPiNexusContract(contract);
    }
  }, [active, library, piNexusContract]);

  useEffect(() => {
    if (piNexusContract) {
      piNexusContract.methods.getGovernanceProposals().call().then(proposals => {
        setGovernanceProposals(proposals);
      });
      piNexusContract.methods.getRewards().call().then(rewards => {
        setRewards(rewards);
      });
      piNexusContract.methods.getLiquidity().call().then(liquidity => {
        setLiquidity(liquidity);
      });
      piNexusContract.methods.getBorrowingRequests().call().then(requests => {
        setBorrowingRequests(requests);
      });
      piNexusContract.methods.getChartData().call().then(data => {
        setChartData(data);
      });
    }
  }, [piNexusContract]);

  const voteOnGovernanceProposal = (proposalId, vote) => {
    piNexusContract.methods.voteOnGovernanceProposal(proposalId, vote).send({ from: account });
    notify(`Voted on proposal ${proposalId} with ${vote ? 'in favor' : 'against'}`);
  };

  const distributeRewards = () => {
    piNexusContract.methods.distributeRewards().send({ from: account });
    notify('Rewards distributed successfully');
  };

  const addLiquidity = (amount) => {
    piNexusContract.methods.addLiquidity(amount).send({ from: account, value: amount });
    notify(`Added ${amount} liquidity to the pool`);
  };

  const requestBorrowing = (amount) => {
    piNexusContract.methods.requestBorrowing(amount).send({ from: account });
    notify(`Requested ${amount} borrowing`);
  };

  const approveBorrowingRequest = (requestId) => {
    piNexusContract.methods.approveBorrowingRequest(requestId).send({ from: account });
    notify(`Approved borrowing request ${requestId}`);
  };

  const handleChartClick = (event) => {
    const { x, y } = event.nativeEvent.offsetX;
    const dataPoint = chartRef.current.getDatasetAtEvent(event)[0];
    if (dataPoint) {
      toast(`Clicked on data point: ${dataPoint.label} - ${dataPoint.value}`);
    }
  };

  useInterval(() => {
    piNexusContract.methods.getChartData().call().then(data => {
      setChartData(data);
    });
  }, 10000);

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="Dashboard">
      <div className="Governance">
        <h2>Governance</h2>
        <ul>
          {governanceProposals.map((proposal, index) => (
            <li key={index}>
              <h3>{proposal.description}</h3>
              <p>Status: {proposal.status}</p>
              <button onClick={() => voteOnGovernanceProposal(index, true)}>Approve</button>
              <button onClick={() => voteOnGovernanceProposal(index, false)}>Reject</button>
            </li>
          ))}
        </ul>
        <button onClick={distributeRewards}>Distribute Rewards</button>
      </div>
      <div className="Liquidity">
        <h2>Liquidity</h2>
        <p>Total liquidity: {liquidity} wei</p>
        <button onClick={() => addLiquidity(100000000000000000)}>Add Liquidity</button>
      </div>
      <div className="Borrowing">
        <h2>Borrowing</h2>
        <ul>
          {borrowingRequests.map((request, index) => (
            <li key={index}>
              <h3>{request.amount} wei</h3>
              <p>Status: {request.status}</p>
              <button onClick={() => approveBorrowingRequest(index)}>Approve</button>
            </li>
          ))}
        </ul>
        <button onClick={() => requestBorrowing(100000000000000000)}>Request Borrowing</button>
      </div>
      <div className="Chart">
        <h2>Liquidity Pool</h2>
        <Line ref={chartRef} data={chartData} onClick={handleChartClick} />
      </div>
    </div>
  );
};

export default Dashboard;
