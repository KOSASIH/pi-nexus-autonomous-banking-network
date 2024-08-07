import React, { useState, useEffect } from 'react';
import { Dialogflow } from '@google-cloud/dialogflow';
import { useApi } from '../context/api';

const ChatWindow = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const api = useApi();

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

  const handleSendMessage = () => {
    const message = inputValue.trim();
    if (message) {
      setInputValue('');
      handleSendMessage(message);
    }
  };

  const handleGetBalance = async () => {
    const balance = await api.getPiBalance();
    setMessages((prevMessages) => [...prevMessages, `Your balance is ${balance} Pi coins`]);
  };

  const handleSendTransaction = async () => {
    const amount = 10; // hardcoded for demo purposes
    const recipient = 'recipient_address'; // hardcoded for demo purposes
    const transaction = await api.sendPiTransaction(amount, recipient);
    setMessages((prevMessages) => [...prevMessages, `Transaction sent: ${transaction}`]);
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
    </div>
  );
};

export default ChatWindow;
