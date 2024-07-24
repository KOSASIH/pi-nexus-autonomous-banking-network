import { useState, useEffect } from 'react';
import Web3 from 'web3';
import { WalletConnect } from '@walletconnect/client';

const useWalletConnect = () => {
  const [walletConnected, setWalletConnected] = useState(false);
  const [walletAddress, setWalletAddress] = useState('');
  const [walletBalance, setWalletBalance] = useState(0);

  useEffect(() => {
    const walletConnect = new WalletConnect({
      bridge: 'https://bridge.walletconnect.org',
      qrcodeModal: {
        open: async (uri) => {
          console.log('Open QR Code Modal');
          console.log(uri);
        },
        close: () => {
          console.log('Close QR Code Modal');
        },
      },
    });

    walletConnect.on('connect', (error, payload) => {
      if (error) {
        console.error(error);
      } else {
        setWalletConnected(true);
        setWalletAddress(payload.accounts[0]);
      }
    });

    walletConnect.on('disconnect', () => {
      setWalletConnected(false);
      setWalletAddress('');
    });

    walletConnect.on('session_update', (error, payload) => {
      if (error) {
        console.error(error);
      } else {
        setWalletBalance(payload.accounts[0].balance);
      }
    });
  }, []);

  const connectWallet = async () => {
    try {
      await walletConnect.connect();
    } catch (error) {
      console.error(error);
    }
  };

  const disconnectWallet = async () => {
    try {
      await walletConnect.disconnect();
    } catch (error) {
      console.error(error);
    }
  };

  return {
    walletConnected,
    walletAddress,
    walletBalance,
    connectWallet,
    disconnectWallet,
  };
};

export default useWalletConnect;
