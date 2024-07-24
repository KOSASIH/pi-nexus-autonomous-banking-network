import React, { useState } from 'react';
import { PiBrowser } from '@pi-network/pi-browser-sdk';

const PiBrowserWallet = () => {
  const [walletBalance, setWalletBalance] = useState(0);
  const [transactionHistory, setTransactionHistory] = useState([]);
  const [sendAmount, setSendAmount] = useState(0);
  const [receiveAddress, setReceiveAddress] = useState('');

  useEffect(() => {
    // Fetch wallet balance
    PiBrowser.getWalletBalance().then(balance => setWalletBalance(balance));

    // Fetch transaction history
    PiBrowser.getTransactionHistory().then(history => setTransactionHistory(history));
  }, []);

  const handleSend = async () => {
    // Send PI to recipient
    await PiBrowser.sendPI(sendAmount, receiveAddress);
  };

  return (
    <div>
      <h1>Pi Browser Wallet</h1>
      <section>
        <h2>Wallet Balance</h2>
        <p>{walletBalance} PI</p>
      </section>
      <section>
        <h2>Transaction History</h2>
        <ul>
          {transactionHistory.map(transaction => (
            <li key={transaction.id}>{transaction.date} - {transaction.amount} {transaction.symbol}</li>
          ))}
        </ul>
      </section>
      <section>
        <h2>Send PI</h2>
        <input
          type="number"
          value={sendAmount}
          onChange={e => setSendAmount(e.target.value)}
          placeholder="Enter amount"
        />
        <input
          type="text"
          value={receiveAddress}
          onChange={e => setReceiveAddress(e.target.value)}
          placeholder="Enter recipient address"
        />
        <button onClick={handleSend}>Send</button>
      </section>
    </div>
  );
};

export default PiBrowserWallet;
