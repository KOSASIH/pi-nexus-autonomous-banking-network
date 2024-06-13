importHead from 'next/head';
import { useState, useEffect } from 'eact';
import Web3 from 'web3';
import { SpectraSyndicate } from '../contracts/SpectraSyndicate.sol';

const Home = () => {
  const [stakeholders, setStakeholders] = useState([]);
  const [tokenBalance, setTokenBalance] = useState(0);
  const [reputationScore, setReputationScore] = useState(0);

  useEffect(() => {
    // Initialize Web3 and contract instance
    const web3 = new Web3(window.ethereum);
    const contract = new SpectraSyndicate(web3);

    // Fetch stakeholders and token balances
    contract.getStakeholders().then((stakeholders) => setStakeholders(stakeholders));
    contract.getTokenBalance().then((balance) => setTokenBalance(balance));

    // Fetch reputation score
    contract.getReputationScore().then((score) => setReputationScore(score));
  }, []);

  return (
    <div>
      <Head>
        <title>SpectraSyndicate</title>
      </Head>
      <h1>Welcome to SpectraSyndicate!</h1>
      <p>Stakeholders: {stakeholders.length}</p>
      <p>Token Balance: {tokenBalance}</p>
      <p>Reputation Score: {reputationScore}</p>
    </div>
  );
};

export default Home;
