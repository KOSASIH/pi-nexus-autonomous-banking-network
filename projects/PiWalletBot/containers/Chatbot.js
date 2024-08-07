import React, { useState, useEffect } from 'react';
import { useApi } from '../context/api';
import { useAuth } from '../context/auth';
import ChatWindow from '../components/ChatWindow';

const Chatbot = () => {
  const api = useApi();
  const auth = useAuth();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const getMessages = async () => {
      try {
        const response = await api.getMessages(auth.user.address);
        setMessages(response.data);
      } catch (error) {
        console.error(error);
      } finally {
        setLoading(false);
      }
    };

    getMessages();
  }, []);

  const handleSendMessage = async () => {
    try {
      const message = inputValue.trim();
      if (message) {
        setInputValue('');
        const response = await api.sendMessage(auth.user.address, message);
        setMessages((prevMessages) => [...prevMessages, response.data]);
      }
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="chatbot-container">
      <h2>Chatbot</h2>
      <ChatWindow
        messages={messages}
        onSendMessage={handleSendMessage}
        inputValue={inputValue}
        onChange={(event) => setInputValue(event.target.value)}
      />
    </div>
  );
};

export default Chatbot;
