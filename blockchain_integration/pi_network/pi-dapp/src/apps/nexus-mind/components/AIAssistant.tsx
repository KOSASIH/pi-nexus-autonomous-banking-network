import React, { useState, useEffect } from 'react';
import { Dialogflow } from 'dialogflow';
import { Wit } from 'wit.ai';

interface AIAssistantProps {
  userQuery: string;
}

const AIAssistant: React.FC<AIAssistantProps> = ({ userQuery }) => {
  const [response, setResponse] = useState('');
  const [intent, setIntent] = useState('');

  useEffect(() => {
    const dialogflow = new Dialogflow();
    const wit = new Wit();

    dialogflow.detectIntent(userQuery).then((response) => {
      setResponse(response.fulfillmentText);
      setIntent(response.intent);
    });

    wit.message(userQuery).then((data) => {
      setResponse(data.entities.intent[0].value);
      setIntent(data.entities.intent[0].value);
    });
  }, [userQuery]);

  return (
    <div>
      <h2>AI Assistant</h2>
      <p>Intent: {intent}</p>
      <p>Response: {response}</p>
    </div>
  );
};

export default AIAssistant;
