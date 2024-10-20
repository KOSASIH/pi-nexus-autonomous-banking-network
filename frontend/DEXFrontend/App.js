import React, { useState, useEffect } from 'react';
import Trade from './Trade';

const App = () => {
  const [trades, setTrades] = useState([]);
  const [tradeCount, setTradeCount] = useState(0);

  useEffect(() => {
    // Get the number of trades
    getTradeCount().then((count) => {
      setTradeCount(count);
    });

    // Get the trade details
    for (let i = 0; i < tradeCount; i++) {
      getTrade(i).then((trade) => {
        setTrades((prevTrades) => [...prevTrades, trade]);
      });
    }
  }, []);

  return (
    <div>
      <h1>Decentralized Exchange (DEX)</h1>
      <ul>
        {trades.map((trade, index) => (
          <li key={index}>
            <Trade trade={trade} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default App;
