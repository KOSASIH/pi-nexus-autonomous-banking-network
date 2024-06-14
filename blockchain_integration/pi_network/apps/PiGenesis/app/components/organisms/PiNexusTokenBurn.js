import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';

const PiNexusTokenBurn = () => {
  const [burnData, setBurnData] = useState(null);
  const { blockchain } = useBlockchain();

  useEffect(() => {
    const fetchBurnData = async () => {
      const data = await blockchain.getBurnData();
     setBurnData(data);
    };

    fetchBurnData();
  }, [blockchain]);

  const handleBurn = async (amount) => {
    const burnResult = await blockchain.burnTokens(amount);
    setBurnData((prevData) => ({
     ...prevData,
      burnedTokens: prevData.burnedTokens + burnResult.burnedTokens,
    }));
  };

  return (
    <div>
      <h1>Pi Nexus Token Burn</h1>
      {burnData && (
        <BurnDataViewer data={burnData} />
      )}
      <BurnForm onSubmit={handleBurn} />
    </div>
  );
};

export default PiNexusTokenBurn;
