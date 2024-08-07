import React, { useState, useEffect } from 'react';
import { Dialogflow } from '@google-cloud/dialogflow';
import { useApi } from '../context/api';
import { useAuth } from '../context/auth';

const ChatWindow = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const api = useApi();
  const auth = useAuth();

  useEffect(() => {
    const dialogflow = new Dialogflow();
    dialogflow.setProjectId('pi-wallet-bot');

    const handleSendMessage = (message) => {
      dialogflow.detectIntent(message, (response) => {
        setMessages((prevMessages) => [...prevMessages, response]);
      });
    };

    return () => {
      handleSendMessage.cancel();
    };
  }, []);

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleSendMessage = async () => {
    const message = inputValue.trim();
    if (message) {
      setInputValue('');
      try {
        setLoading(true);
        const response = await dialogflow.detectIntent(message);
        setMessages((prevMessages) => [...prevMessages, response]);
        setLoading(false);
      } catch (error) {
        setError(error.message);
        setLoading(false);
      }
    }
  };

  const handleGetBalance = async () => {
    try {
      setLoading(true);
      const balance = await api.getPiBalance(auth.user.address);
      setMessages((prevMessages) => [...prevMessages, `Your balance is ${balance} Pi coins`]);
      setLoading(false);
    } catch (error) {
      setError(error.message);
      setLoading(false);
    }
  };

  const handleSendTransaction = async () => {
    try {
      setLoading(true);
      const amount = 10; // hardcoded for demo purposes
      const recipient = 'recipient_address'; // hardcoded for demo purposes
      const transaction = await api.sendPiTransaction(auth.user.address, recipient, amount);
      setMessages((prevMessages) => [...prevMessages, `Transaction sent: ${transaction}`]);
      setLoading(false);
    } catch (error) {
      setError(error.message);
      setLoading(false);
    }
  };

  return (
    <div className="chat-window">
      <h2>Pi Wallet Bot</h2>
      <ul>
        {messages.map((message, index) => (
          <li key={index}>{message}</li>
        ))}
      </ul>
      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        placeholder="Type a message..."
      />
      <button onClick={handleSendMessage}>Send</button>
      <button onClick={handleGetBalance}>Get Balance</button>
      <button onClick={handleSendTransaction}>Send Transaction</button>
      {loading ? <p>Loading...</p> : null}
      {error ? <p>Error: {error}</p> : null}
    </div>
  );
};

export default ChatWindow;
