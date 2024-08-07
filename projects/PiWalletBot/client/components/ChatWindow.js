import React, { useState, useEffect } from 'react';
import { Dialogflow } from '@google-cloud/dialogflow';

const ChatWindow = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');

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
    </div>
  );
};

export default ChatWindow;
