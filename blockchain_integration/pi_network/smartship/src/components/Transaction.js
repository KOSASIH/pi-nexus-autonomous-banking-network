import React, { useState, useEffect } from 'react';
import Web3 from 'web3';
import { useContractLoader, useContractReader } from 'eth-hooks';

const Transaction = ({ transactionId, provider, blockNumber }) => {
  const [sender, setSender] = useState('');
  const [receiver, setReceiver] = useState('');
  const [amount, setAmount] = useState(0);
  const contract = useContractLoader('MyContract', provider);

  useEffect(() => {
    async function fetchData() {
      const transaction = await contract.methods.transactions(transactionId).call();
      setSender(transaction.sender);
      setReceiver(transaction.receiver);
      setAmount(Web3.utils.fromWei(transaction.amount, 'ether'));
    }
    fetchData();
  }, [transactionId, blockNumber]);

  return (
    <div>
      <h2>Transaction: {transactionId}</h2>
      <p>Sender: {sender}</p>
      <p>Receiver: {receiver}</p>
      <p>Amount: {amount} ETH</p>
    </div>
  );
};

export default Transaction;
