import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [trades, setTrades] = useState([]);
  const [newTrade, setNewTrade] = useState({
    buyer: '',
    seller: '',
    product: '',
    quantity: '',
    price: '',
  });

  useEffect(() => {
    axios.get('http://localhost:3000/getTrade')
      .then((response) => {
        setTrades(response.data);
      })
      .catch((error) => {
        console.error(error);
      });
  }, []);

  const handleCreateTrade = (event) => {
    event.preventDefault();
    axios.post('http://localhost:3000/createTrade', newTrade)
      .then((response) => {
        setTrades([...trades, response.data]);
        setNewTrade({
          buyer: '',
          seller: '',
          product: '',
          quantity: '',
          price: '',
        });
      })
      .catch((error) => {
        console.error(error);
      });
  };

  const handleUpdateTrade = (event) => {
    event.preventDefault();
    const tradeId = event.target.id;
    const status = event.target.value;
    axios.put(`http://localhost:3000/updateTrade/${tradeId}`, { status })
      .then((response) => {
        setTrades(trades.map((trade) => {
          if (trade.id === tradeId) {
            trade.status = status;
          }
          return trade;
        }));
      })
      .catch((error) => {
        console.error(error);
      });
  };

  return (
    <div>
      <h1>Supply Chain Finance</h1>
      <form onSubmit={handleCreateTrade}>
        <label>Buyer:</label>
        <input type="text" value={newTrade.buyer} onChange={(event) => setNewTrade({ ...newTrade, buyer: event.target.value })} />
        <br />
        <label>Seller:</label>
        <input type="text" value={newTrade.seller} onChange={(event) => setNewTrade({ ...newTrade, seller: event.target.value })} />
        <br />
        <label>Product:</label>
        <input type="text" value={newTrade.product} onChange={(event) => setNewTrade({ ...newTrade, product: event.target.value })} />
        <br />
        <label>Quantity:</label>
        <input type="text" value={newTrade.quantity} onChange={(event) => setNewTrade({ ...newTrade, quantity: event.target.value })} />
        <br />
        <label>Price:</label>
        <input type="text" value={newTrade.price} onChange={(event) => setNewTrade({ ...newTrade, price: event.target.value })} />
        <br />
        <button type="submit">Create Trade</button>
      </form>
      <h2>Trades</h2>
      <ul>
        {trades.map((trade) => (
          <li key={trade.id}>
            <span>Buyer: {trade.buyer}</span>
            <span>Seller: {trade.seller}</span>
            <span>Product: {trade.product}</span>
            <span>Quantity: {trade.quantity}</span>
            <span>Price: {trade.price}</span>
            <span>Status: {trade.status}</span>
            <button id={trade.id} value="pending" onClick={handleUpdateTrade}>Update Status</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
