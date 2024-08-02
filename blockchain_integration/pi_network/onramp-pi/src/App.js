// onramp-pi/src/App.js

import React, { useState, useEffect } from 'eact';
import { BrowserRouter, Route, Switch } from 'eact-router-dom';
import { Web3Provider } from '@ethersproject/providers';
import { useWeb3React } from '@web3-react/core';
import { WalletDashboard } from './components/WalletDashboard';
import { TransactionList } from './components/TransactionList';
import { FiatOnRamp } from './components/FiatOnRamp';
import { Web3Utils } from './utils/web3';
import { config } from './config';

const App = () => {
  const [walletConnected, setWalletConnected] = useState(false);
  const [walletAddress, setWalletAddress] = useState('');
  const [walletBalance, setWalletBalance] = useState(0);
  const [erc20Balance, setERC20Balance] = useState(0);
  const [uniswapV2Quote, setUniswapV2Quote] = useState(null);

  const { active, account, library } = useWeb3React();

  useEffect(() => {
    if (active && account) {
      setWalletConnected(true);
      setWalletAddress(account);
      Web3Utils.getWalletBalance(account).then((balance) => setWalletBalance(balance));
      Web3Utils.getERC20Balance(account, config.erc20TokenAddress).then((balance) => setERC20Balance(balance));
    }
  }, [active, account]);

  const handleConnectWallet = async () => {
    try {
      await library.send('eth_requestAccounts', []);
      setWalletConnected(true);
    } catch (error) {
      console.error(error);
    }
  };

  const handleDisconnectWallet = async () => {
    try {
      await library.send('eth_disconnect', []);
      setWalletConnected(false);
    } catch (error) {
      console.error(error);
    }
  };

  const handleSwapETHForERC20 = async (amount) => {
    try {
      const txHash = await Web3Utils.swapETHForERC20(amount, config.erc20TokenAddress);
      console.log(`Swap ETH for ERC20 successful: ${txHash}`);
    } catch (error) {
      console.error(error);
    }
  };

  const handleGetUniswapV2Quote = async (amount) => {
    try {
      const quote = await Web3Utils.getUniswapV2RouterQuote(amount, config.erc20TokenAddress);
      setUniswapV2Quote(quote);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <BrowserRouter>
      <Switch>
        <Route exact path="/" render={() => (
          <WalletDashboard
            walletConnected={walletConnected}
            walletAddress={walletAddress}
            walletBalance={walletBalance}
            erc20Balance={erc20Balance}
            onConnectWallet={handleConnectWallet}
            onDisconnectWallet={handleDisconnectWallet}
          />
        )} />
        <Route path="/transactions" render={() => (
          <TransactionList walletAddress={walletAddress} />
        )} />
        <Route path="/fiat-onramp" render={() => (
          <FiatOnRamp
            walletAddress={walletAddress}
            erc20Balance={erc20Balance}
            uniswapV2Quote={uniswapV2Quote}
            onSwapETHForERC20={handleSwapETHForERC20}
            onGetUniswapV2Quote={handleGetUniswapV2Quote}
          />
        )} />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
