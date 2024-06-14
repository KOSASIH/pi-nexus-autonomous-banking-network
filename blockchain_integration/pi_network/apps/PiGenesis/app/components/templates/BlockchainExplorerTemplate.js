import { useState, useEffect } from 'react';
import { useWeb3 } from '@web3/web3-react';

const BlockchainExplorerTemplate = () => {
  const [blockchainData, setBlockchainData] = useState(null);
  const { web3 } = useWeb3();

  useEffect(() => {
    const fetchBlockchainData = async () => {
      const data = await web3.eth.getBlockchainData();
      setBlockchainData(data);
    };

    fetchBlockchainData();
  }, [web3]);

  return (
    <div>
      <h1>Pi Nexus Blockchain Explorer</h1>
      {blockchainData && (
        <BlockchainDataViewer data={blockchainData} />
      )}
    </div>
  );
};

export default BlockchainExplorerTemplate;
