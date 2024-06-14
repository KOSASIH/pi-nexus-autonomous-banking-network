import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';

const PiNexusTokenStaking = () => {
  const [stakingData, setStakingData] = useState(null);
  const { blockchain } = useBlockchain();

  useEffect(() => {
    const fetchStakingData = async () => {
      const data = await blockchain.getStakingData();
      setStakingData(data);
    };

    fetchStakingData();
  }, [blockchain]);

  const handleStake = async (amount) => {
    const stakingResult = await blockchain.stakeTokens(amount);
    setStakingData((prevData) => ({
      ...prevData,
      stakedTokens: prevData.stakedTokens + stakingResult.stakedTokens,
    }));
  };

  return (
    <div>
      <h1>Pi Nexus Token Staking</h1>
      {stakingData && (
        <StakingDataViewer data={stakingData} />
      )}
      <StakeForm onSubmit={handleStake} />
    </div>
  );
};

export default PiNexusTokenStaking;
