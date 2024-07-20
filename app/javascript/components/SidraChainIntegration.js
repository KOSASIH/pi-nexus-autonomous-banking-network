// app/javascript/components/SidraChainIntegration.js
import React, { useState, useEffect } from 'react';
import { useSpree } from 'spree-react';

const SidraChainIntegration = () => {
  const [sidraChainData, setSidraChainData] = useState({});

  useEffect(() => {
    // Fetch Sidra Chain data using Spree API
    fetch('/api/v2/sidra_chain_data')
      .then(response => response.json())
      .then(data => setSidraChainData(data));
  }, []);

  return (
    <div>
      <h1>Sidra Chain Integration</h1>
      <p>Data: {sidraChainData}</p>
    </div>
  );
};

export default SidraChainIntegration;
