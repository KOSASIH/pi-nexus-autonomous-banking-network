import React, { useState, useEffect } from 'eact';
import Web3 from 'web3';
import { PiNetworkRouter } from './PiNetworkRouter';
import './PiNetworkUI.css';

const PiNetworkUI = () => {
  const [fromAddress, setFromAddress] = useState('');
  const [toAddress, setToAddress] = useState('');
  const [amount, setAmount] = useState(0);
  const [txHash, setTxHash] = useState('');

  useEffect(() => {
    async function getTxHash() {
      const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));
      const piNetworkRouterAddress = '0x...'; // Replace with the deployed PiNetworkRouter address
      const piNetworkRouter = new PiNetworkRouter(piNetworkRouterAddress, web3);

      const txCount = await web3.eth.getTransactionCount(fromAddress);
      const tx = {
        from: fromAddress,
        to: piNetworkRouterAddress,
        value: web3.utils.toWei(amount, 'ether'),
        gas: '20000',
        gasPrice: web3.utils.toWei('20', 'gwei'),
        nonce: txCount
      };

      const signedTx = await web3.eth.accounts.signTransaction(tx, '0x...'); // Replace with the sender's private key
      const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

      setTxHash(receipt.transactionHash);
    }

    getTxHash();
  }, [fromAddress, toAddress, amount]);

  const handleFromAddressChange = (event) => {
    setFromAddress(event.target.value);
  };

  const handleToAddressChange = (event) => {
    setToAddress(event.target.value);
  };

  const handleAmountChange = (event) => {
    setAmount(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    piNetworkRouter.transfer(fromAddress, toAddress, amount);
  };

  return (
    <div>
      <h1>Pi Network UI</h1>
      <form onSubmit={handleSubmit}>
        <label>From Address:</label>
        <input type="text" value={fromAddress} onChange={handleFromAddressChange} />
        <br />
        <label>To Address:</label>
        <input type="text" value={toAddress} onChange={handleToAddressChange} />
        <br />
        <label>Amount:</label>
        <input type="number" value={amount} onChange={handleAmountChange} />
        <br />
        <button type="submit">Transfer</button>
      </form>
      <p>Transaction Hash: {txHash}</p>
    </div>
  );
};

export default PiNetworkUI;
