import React, { useState, useEffect, useContext } from "react";
import Web3 from "web3";
import { useWeb3React } from "@web3-react/core";
import { Contract } from "web3/eth/contract";
import PI_Nexus_ABI from "../../../contracts/PI_Nexus_Autonomous_Banking_Network_v3.json";
import { ThemeContext } from "../../contexts/ThemeContext";
import { WalletContext } from "../../contexts/WalletContext";
import { NotificationContext } from "../../contexts/NotificationContext";
import "./PI_Nexus_Dashboard.css";
import { Chart } from "react-chartjs-2";
import { LineChart, Line, CartesianGrid, XAxis, YAxis } from "recharts";
import { useMediaQuery } from "react-responsive";

const PI_NEXUS_CONTRACT_ADDRESS = "0x..."; // Replace with actual contract address

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
  const [isMobile] = useMediaQuery({ query: "(max-width: 768px)" });

  useEffect(() => {
    if (active && library && !piNexusContract) {
      const web3 = new Web3(library.provider);
      const contract = new Contract(
        PI_NEXUS_CONTRACT_ADDRESS,
        PI_Nexus_ABI,
        web3.currentProvider,
      );
      setPiNexusContract(contract);
    }
  }, [active, library, piNexusContract]);

  useEffect(() => {
    if (piNexusContract) {
      piNexusContract.methods
        .getGovernanceProposals()
        .call()
        .then((proposals) => {
          setGovernanceProposals(proposals);
        });
      piNexusContract.methods
        .getRewards()
        .call()
        .then((rewards) => {
          setRewards(rewards);
        });
      piNexusContract.methods
        .getLiquidity()
        .call()
        .then((liquidity) => {
          setLiquidity(liquidity);
        });
      piNexusContract.methods
        .getBorrowingRequests()
        .call()
        .then((requests) => {
          setBorrowingRequests(requests);
        });
      piNexusContract.methods
        .getChartData()
        .call()
        .then((data) => {
          setChartData(data);
        });
    }
  }, [piNexusContract]);

  const voteOnGovernanceProposal = (proposalId, vote) => {
    piNexusContract.methods
      .voteOnGovernanceProposal(proposalId, vote)
      .send({ from: account });
    notify(
      `Voted on proposal ${proposalId} with ${vote ? "in favor" : "against"}`,
    );
  };

  const distributeRewards = () => {
    piNexusContract.methods.distributeRewards().send({ from: account });
    notify("Rewards distributed successfully");
  };

  const addLiquidity = (amount) => {
    piNexusContract.methods
      .addLiquidity(amount)
      .send({ from: account, value: amount });
    notify(`Added ${amount} liquidity to the pool`);
  };

  const requestBorrowing = (amount) => {
    piNexusContract.methods.requestBorrowing(amount).send({ from: account });
    notify(`Requested ${amount} borrowing`);
  };

  const approveBorrowingRequest = (requestId) => {
    piNexusContract.methods
      .approveBorrowingRequest(requestId)
      .send({ from: account });
    notify(`Approved borrowing request ${requestId}`);
  };

  return (
    <div className={`PI_Nexus_Dashboard ${theme}`}>
      <h1>PI Nexus Autonomous Banking Network</h1>
      <h2>Governance Proposals</h2>
      {governanceProposals.map((proposal) => (
        <div key={proposal[0]} className="proposal">
          <p>ID: {proposal[0]}</p>
          <p>Description: {proposal[1]}</p>
          <p>Votes: {proposal[2]}</p>
          <button onClick={() => voteOnGovernanceProposal(proposal[0], true)}>
            Vote In Favor
          </button>
          <button onClick={() => voteOnGovernanceProposal(proposal[0], false)}>
            Vote Against
          </button>
        </div>
      ))}
      <h2>Rewards</h2>
      {rewards.map((reward) => (
        <div key={reward[0]} className="reward">
          <p>Address: {reward[0]}</p>
          <p>Amount: {reward[1]}</p>
        </div>
      ))}
      <h2>Liquidity</h2>
      <p>Liquidity: {liquidity}</p>
      <button onClick={distributeRewards}>Distribute Rewards</button>
      <button onClick={() => addLiquidity(100)}>Add Liquidity</button>
      <h2>Borrowing Requests</h2>
      {borrowingRequests.map((request) => (
        <div key={request[0]} className="request">
          <p>ID: {request[0]}</p>
          <p>Address: {request[1]}</p>
          <p>Amount: {request[2]}</p>
          <button onClick={() => approveBorrowingRequest(request[0])}>
            Approve
          </button>
        </div>
      ))}
      <h2>Chart</h2>
      {isMobile ? (
        <LineChart width={300} height={200} data={chartData}>
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
          <CartesianGrid stroke="#ccc" />
          <XAxis dataKey="date" />
          <YAxis />
        </LineChart>
      ) : (
        <Chart type="line" data={chartData} />
      )}
    </div>
  );
};

export default PI_Nexus_Dashboard;
