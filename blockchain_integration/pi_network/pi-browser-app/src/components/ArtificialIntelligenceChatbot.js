import React, { useState, useEffect } from 'eact';
import { ArtificialIntelligenceAPI } from '../api';

const ArtificialIntelligenceChatbot = () => {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (event) => {
    setInput(event.target.value);
  };

  const handleSendMessage = async () => {
    setLoading(true);
    const response = await ArtificialIntelligenceAPI.sendMessage(input);
    setResponse(response.data);
    setLoading(false);
  };

  return (
    <div className="artificial-intelligence-chatbot">
      <h1>Artificial Intelligence Chatbot</h1>
      <input type="text" value={input} onChange={handleInputChange} placeholder="Type a message..." />
      <button onClick={handleSendMessage}>Send</button>
      {loading? (
        <p>Loading...</p>
      ) : (
        <p>{response}</p>
      )}
    </div>
  );
};

export default ArtificialIntelligenceChatbot;
