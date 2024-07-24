import React, { useState, useEffect } from 'react';
import { PiBrowser } from '@pi-network/pi-browser-sdk';

const PiBrowserDashboard = () => {
  const [cryptocurrencyPrices, setCryptocurrencyPrices] = useState({});
  const [transactionHistory, setTransactionHistory] = useState([]);
  const [walletBalance, setWalletBalance] = useState(0);
  const [newsFeed, setNewsFeed] = useState([]);

  useEffect(() => {
    // Fetch real-time cryptocurrency prices
    PiBrowser.getPriceFeed().then(prices => setCryptocurrencyPrices(prices));

    // Fetch transaction history
    PiBrowser.getTransactionHistory().then(history => setTransactionHistory(history));

    // Fetch wallet balance
    PiBrowser.getWalletBalance().then(balance => setWalletBalance(balance));

    // Fetch news feed
    PiBrowser.getNewsFeed().then(feed => setNewsFeed(feed));
  }, []);

  return (
    <div>
      <h1>Pi Browser Dashboard</h1>
      <section>
        <h2>Cryptocurrency Prices</h2>
        <ul>
          {Object.keys(cryptocurrencyPrices).map(symbol => (
            <li key={symbol}>{symbol}: {cryptocurrencyPrices[symbol]}</li>
          ))}
        </ul>
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
        <h2>Wallet Balance</h2>
        <p>{walletBalance} PI</p>
      </section>
      <section>
        <h2>News Feed</h2>
        <ul>
          {newsFeed.map(article => (
            <li key={article.id}>
              <a href={article.url}>{article.title}</a>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
};

export default PiBrowserDashboard;
