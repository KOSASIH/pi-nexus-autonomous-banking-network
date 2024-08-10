import React from 'react';

const NodeCard = ({ node }) => {
  return (
    <div className="node-card">
      <h2>{node.name}</h2>
      <p>Reputation: {node.reputation}</p>
      <p>Incentivization: {node.incentivization}</p>
    </div>
  );
};

export default NodeCard;
