import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';

const PiNexusWalletDashboard = () => {
  const [walletData, setWalletData] = useState(null);
  const { blockchain } = useBlockchain();

  useEffect(() => {
    const fetchWalletData = async () => {
      const data = await blockchain.getWalletData();
      setWalletData(data);
    };

    fetchWalletData();
  }, [blockchain]);

  return (
    <div>
      <h1>Pi Nexus Wallet Dashboard</h1>
      {walletData && (
        <WalletDataViewer data={walletData} />
      )}
    </div>
  );
};

export default PiNexusWalletDashboard;
