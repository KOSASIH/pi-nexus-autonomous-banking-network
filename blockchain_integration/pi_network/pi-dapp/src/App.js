import React from 'react';
import { useWeb3React } from '@web3-react/core';
import { usePiNetwork } from './PiNetworkProvider';

const App = () => {
  const { account, library } = useWeb3React();
  const { piTokenBalance, stakingBalance, lendingBalance, governanceVotingPower } = usePiNetwork();

  return (
    <div>
      <h1>Pi-Nexus Autonomous Banking Network</h1>
      <p>Account: {account}</p>
      <p>Pi Token Balance: {piTokenBalance}</p>
      <p>Staking Balance: {stakingBalance}</p>
      <p>Lending Balance: {lendingBalance}</p>
      <p>Governance Voting Power: {governanceVotingPower}</p>
      <button onClick={() => library.sendTransaction('0x...TransactionData...')}>Send Transaction</button>
    </div>
  );
};

export defaultApp;
