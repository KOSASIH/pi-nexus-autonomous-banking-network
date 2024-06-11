import React, { useState, useEffect } from 'eact';
import { useWeb3 } from '../utils/web3';
import { useMachineLearning } from '../utils/machine-learning';

const GalacticWallet = () => {
  const [balance, setBalance] = useState(0);
  const [transactions, setTransactions] = useState([]);
  const web3 = useWeb3();
  const mlModel = useMachineLearning();

  useEffect(() => {
    const fetchBalance = async () => {
      const balance = await web3.eth.getBalance(web3.eth.defaultAccount);
      setBalance(balance);
    };
    fetchBalance();
  }, [web3]);

  useEffect(() => {
    const fetchTransactions = async () => {
      const transactions = await web3.eth.getTransactionCount(web3.eth.defaultAccount);
      setTransactions(transactions);
    };
    fetchTransactions();
  }, [web3]);

  const handleTransaction = async (amount: number, recipient: string) => {
    const transaction = await web3.eth.sendTransaction({
      from: web3.eth.defaultAccount,
      to: recipient,
      value: amount,
    });
    mlModel.train(transaction); // Train the ML model with the transaction data
  };

  return (
    <div>
      <h1>Galactic Wallet</h1>
      <p>Balance: {balance}</p>
      <p>Transactions: {transactions.length}</p>
      <button onClick={() => handleTransaction(1, '0x...')}>Send 1 ETH</button>
    </div>
  );
};

export default GalacticWallet;
