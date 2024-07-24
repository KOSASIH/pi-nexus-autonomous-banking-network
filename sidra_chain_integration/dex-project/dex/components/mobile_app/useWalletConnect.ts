import { useState, useEffect } from 'eact';
import { WalletConnect } from '@walletconnect/client';

const useWalletConnect = () => {
  const [walletConnected, setWalletConnected] = useState(false);
  const [walletConnector, setWalletConnector] = useState<WalletConnect | null>(null);

  useEffect(() => {
    const walletConnector = new WalletConnect({
      bridge: 'https://bridge.walletconnect.org',
      clientId: 'your-client-id',
    });

    setWalletConnector(walletConnector);

    walletConnector.on('connect', () => {
      setWalletConnected(true);
    });

    walletConnector.on('disconnect', () => {
      setWalletConnected(false);
    });
  }, []);

  const connectWallet = async () => {
    if (!walletConnected) {
      try {
        await walletConnector?.connect();
      } catch (error) {
        console.error(error);
      }
    }
  };

  return { connectWallet, walletConnected };
};

export default useWalletConnect;
