// onramp-pi/src/components/FiatOnRampModal.js

import React, { useState, useEffect } from 'react';
import { Modal, ModalHeader, ModalBody, ModalFooter } from 'reactstrap';
import { useWeb3React } from '@web3-react/core';
import { Web3Utils } from '../utils/web3';
import { config } from '../config';
import { useToast } from '../hooks/useToast';
import { useFiatOnRamp } from '../hooks/useFiatOnRamp';

const FiatOnRampModal = () => {
  const { account, library } = useWeb3React();
  const { toast } = useToast();
  const { fiatOnRamp, setFiatOnRamp } = useFiatOnRamp();
  const [amount, setAmount] = useState(0);
  const [quote, setQuote] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  useEffect(() => {
    if (fiatOnRamp) {
      setAmount(fiatOnRamp.amount);
      setQuote(fiatOnRamp.quote);
    }
  }, [fiatOnRamp]);

  const handleGetQuote = async () => {
    try {
      setLoading(true);
      const quote = await Web3Utils.getUniswapV2RouterQuote(amount, config.erc20TokenAddress);
      setQuote(quote);
      setLoading(false);
    } catch (error) {
      setError(error.message);
      setLoading(false);
    }
  };

  const handleSwap = async () => {
    try {
      setLoading(true);
      const txHash = await Web3Utils.swapETHForERC20(amount, config.erc20TokenAddress);
      toast(`Swap successful! Tx Hash: ${txHash}`);
      setLoading(false);
      setModalOpen(false);
    } catch (error) {
      setError(error.message);
      setLoading(false);
    }
  };

  const handleToggleModal = () => {
    setModalOpen(!modalOpen);
  };

  return (
    <Modal isOpen={modalOpen} toggle={handleToggleModal}>
      <ModalHeader toggle={handleToggleModal}>Fiat On-Ramp</ModalHeader>
      <ModalBody>
        <form>
          <label>
            Amount (ETH):
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
            />
          </label>
          <button onClick={handleGetQuote}>Get Quote</button>
          <button onClick={handleSwap}>Swap</button>
        </form>
        {quote && (
          <div>
            <h2>Quote</h2>
            <ul>
              <li>
                <strong>Amount Out:</strong> {quote.amountOut} {config.erc20TokenSymbol}
              </li>
              <li>
                <strong>Price:</strong> {quote.price} ETH/{config.erc20TokenSymbol}
              </li>
            </ul>
          </div>
        )}
        {loading && <p>Loading...</p>}
        {error && <p>Error: {error}</p>}
      </ModalBody>
      <ModalFooter>
        <button onClick={handleToggleModal}>Close</button>
      </ModalFooter>
    </Modal>
  );
};

export default FiatOnRampModal;
