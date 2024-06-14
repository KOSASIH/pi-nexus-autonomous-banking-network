import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';

const PiNexusTokenLending = () => {
  const [lendingData, setLendingData] = useState(null);
  const { blockchain } = useBlockchain();

  useEffect(() => {
    const fetchLendingData = async () => {
      const data = await blockchain.getLendingData();
      setLendingData(data);
    };

    fetchLendingData();
  }, [blockchain]);

  const handleLend = async (amount, interestRate) => {
    const lendingResult = await blockchain.lendTokens(amount, interestRate);
    setLendingData((prevData) => ({
      ...prevData,
      lentTokens: prevData.lentTokens + lendingResult.lentTokens,
    }));
  };

  return (
    <div>
      <h1>Pi Nexus Token Lending</h1>
      {lendingData && (
        <LendingDataViewer data={lendingData} />
      )}
      <LendForm onSubmit={handleLend} />
    </div>
  );
};

export default PiNexusTokenLending;
