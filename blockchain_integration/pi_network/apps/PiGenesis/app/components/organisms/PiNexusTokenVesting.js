import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';

const PiNexusTokenVesting = () => {
  const [vestingData, setVestingData] = useState(null);
  const { blockchain } = useBlockchain();

  useEffect(() => {
    const fetchVestingData = async () => {
      const data = await blockchain.getVestingData();
      setVestingData(data);
    };

    fetchVestingData();
  }, [blockchain]);

  const handleVest = async (amount, vestingPeriod) => {
    const vestingResult = await blockchain.vestTokens(amount, vestingPeriod);
    setVestingData((prevData) => ({
      ...prevData,
      vestedTokens: prevData.vestedTokens + vestingResult.vestedTokens,
    }));
  };

  return (
    <div>
      <h1>Pi Nexus Token Vesting</h1>
      {vestingData && (
        <VestingDataViewer data={vestingData} />
      )}
      <VestForm onSubmit={handleVest} />
    </div>
  );
};

export default PiNexusTokenVesting;
