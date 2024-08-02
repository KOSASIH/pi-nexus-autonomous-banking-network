// onramp-pi/src/hooks/useFiatOnRamp.js

import { useState, useEffect } from 'react';
import { Web3Utils } from '../utils/web3';
import { config } from '../config';
import { useWeb3React } from '@web3-react/core';

const useFiatOnRamp = () => {
  const { account, library } = useWeb3React();
  const [fiatOnRamp, setFiatOnRamp] = useState({
    amount: 0,
    quote: null,
    erc20Balance: 0,
  });

  useEffect(() => {
    const fetchERC20Balance = async () => {
      const erc20Balance = await Web3Utils.getERC20Balance(config.erc20TokenAddress, account);
      setFiatOnRamp((prev) => ({ ...prev, erc20Balance }));
    };

    fetchERC20Balance();
  }, [account, library]);

  const updateAmount = (amount) => {
    setFiatOnRamp((prev) => ({ ...prev, amount }));
  };

  const getQuote = async () => {
    try {
      const quote = await Web3Utils.getUniswapV2RouterQuote(fiatOnRamp.amount, config.erc20TokenAddress);
      setFiatOnRamp((prev) => ({ ...prev, quote }));
    } catch (error) {
      console.error(error);
    }
  };

  const swap = async () => {
    try {
      const txHash = await Web3Utils.swapETHForERC20(fiatOnRamp.amount, config.erc20TokenAddress);
      console.log(`Swap successful! Tx Hash: ${txHash}`);
    } catch (error) {
      console.error(error);
    }
  };

  return {
    fiatOnRamp,
    updateAmount,
    getQuote,
    swap,
  };
};

export default useFiatOnRamp;
