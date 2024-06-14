import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';

const PiNexusTokenSwap = () => {
  const [swapData, setSwapData] = useState(null);
  const { blockchain } = useBlockchain();

  useEffect(() => {
    const fetchSwapData = async () => {
      const data = await blockchain.getSwapData();
      setSwapData(data);
    };

    fetchSwapData();
  }, [blockchain]);

  const handleSwap = async (amount, targetCurrency) => {
    const swapResult = await blockchain.swapTokens(amount, targetCurrency);
    setSwapData((prevData) => ({
      ...prevData,
      swapHistory: [...prevData.swapHistory, swapResult],
    }));
  };

  return (
    <div>
      <h1>Pi Nexus Token Swap</h1>
      {swapData && (
        <SwapDataViewer data={swapData} />
      )}
      <SwapForm onSubmit={handleSwap} />
    </div>
  );
};

export default PiNexusTokenSwap;
