import { useState, useEffect, useCallback, useMemo } from 'react';
import { ethers } from 'ethers';
import { Web3Provider } from '@ethersproject/providers';
import { useWeb3React } from '@web3-react/core';
import { InjectedConnector } from '@web3-react/injected-connector';
import { WalletConnectConnector } from '@web3-react/walletconnect-connector';
import { LedgerConnector } from '@web3-react/ledger-connector';
import { TrezorConnector } from '@web3-react/trezor-connector';
import { useToast } from '@chakra-ui/react';

const supportedChains = [
  {
    chainId: 1,
    chainName: 'Ethereum',
    rpcUrls: ['https://mainnet.infura.io/v3/YOUR_PROJECT_ID'],
  },
  {
    chainId: 56,
    chainName: 'Binance Smart Chain',
    rpcUrls: ['https://bsc-dataseed.binance.org/'],
  },
  // Add more chains as needed
];

const useWallet = () => {
  const [wallet, setWallet] = useState(null);
  const [account, setAccount] = useState(null);
  const [chainId, setChainId] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  const web3React = useWeb3React();
  const { activate, deactivate } = web3React;

  const injectedConnector = new InjectedConnector({
    supportedChainIds: supportedChains.map((chain) => chain.chainId),
  });

  const walletConnectConnector = new WalletConnectConnector({
    bridge: 'https://bridge.walletconnect.org',
    qrcode: true,
  });

  const ledgerConnector = new LedgerConnector({
    chainId: supportedChains[0].chainId,
    url: supportedChains[0].rpcUrls[0],
  });

  const trezorConnector = new TrezorConnector({
    chainId: supportedChains[0].chainId,
    url: supportedChains[0].rpcUrls[0],
  });

  const connectWallet = useCallback(async () => {
    try {
      setLoading(true);
      const connector = await injectedConnector.connect();
      const provider = new Web3Provider(connector);
      const accounts = await provider.listAccounts();
      const account = accounts[0];
      const chainId = await provider.getNetwork().then((network) => network.chainId);
      setWallet(connector);
      setAccount(account);
      setChainId(chainId);
      toast({
        title: 'Wallet connected',
        description: `Connected to ${connector.name}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      setError(error);
      toast({
        title: 'Error connecting wallet',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  }, [injectedConnector, toast]);

  const disconnectWallet = useCallback(async () => {
    try {
      setLoading(true);
      await deactivate();
      setWallet(null);
      setAccount(null);
      setChainId(null);
      toast({
        title: 'Wallet disconnected',
        description: 'Disconnected from wallet',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      setError(error);
      toast({
        title: 'Error disconnecting wallet',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  }, [deactivate, toast]);

  const switchChain = useCallback(async (chainId) => {
    try {
      setLoading(true);
      await web3React.activate(chainId);
      setChainId(chainId);
      toast({
        title: 'Chain switched',
        description: `Switched to chain ${chainId}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      setError(error);
      toast({
        title: 'Error switching chain',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  }, [web3React, toast]);

    useEffect(() => {
    if (web3React.active) {
      const provider = new Web3Provider(web3React.library);
      const accounts = provider.listAccounts();
      const account = accounts[0];
      const chainId = provider.getNetwork().then((network) => network.chainId);
      setWallet(web3React.library);
      setAccount(account);
      setChainId(chainId);
    }
  }, [web3React]);

  useEffect(() => {
    if (error) {
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  }, [error, toast]);

  const value = useMemo(() => ({
    wallet,
    account,
    chainId,
    connectWallet,
    disconnectWallet,
    switchChain,
    error,
    loading,
  }), [
    wallet,
    account,
    chainId,
    connectWallet,
    disconnectWallet,
    switchChain,
    error,
    loading,
  ]);

  return value;
};

export default useWallet;
   
