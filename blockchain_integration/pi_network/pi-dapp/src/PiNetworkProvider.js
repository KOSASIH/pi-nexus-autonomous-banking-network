import { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import { PiNetworkContract } from './PiNetworkContract';

const PiNetworkProvider = ({ children, web3Provider }) => {
  const [piTokenBalance, setPiTokenBalance] = useState(0);
  const [stakingBalance, setStakingBalance] = useState(0);
  const [lendingBalance, setLendingBalance] = useState(0);
  const [governanceVotingPower, setGovernanceVotingPower] = useState(0);

  useEffect(() => {
    const piNetworkContract = new PiNetworkContract(web3Provider);
    piNetworkContract.getPiTokenBalance().then((balance) => setPiTokenBalance(balance));
    piNetworkContract.getStakingBalance().then((balance) => setStakingBalance(balance));
    piNetworkContract.getLendingBalance().then((balance) => setLendingBalance(balance));
    piNetworkContract.getGovernanceVotingPower().then((power) => setGovernanceVotingPower(power));
  }, [web3Provider]);

  return (
    <div>
      {children}
      <PiTokenBalance balance={piTokenBalance} />
      <StakingBalance balance={stakingBalance} />
      <LendingBalance balance={lendingBalance} />
      <GovernanceVotingPower power={governanceVotingPower} />
    </div>
  );
};

export default PiNetworkProvider;
