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
        <input type="text" value={newTrade
