import React, { useState, useEffect } from 'react';
import { Polkadot } from 'polkadot-js';
import { CrossChainInteroperabilityAPI } from '../api/cross-chain-interoperability';

interface CrossChainInteroperabilityProps {
  user: any;
}

const CrossChainInteroperability: React.FC<CrossChainInteroperabilityProps> = ({ user }) => {
  const [crossChainTransaction, setCrossChainTransaction] = useState({});

  useEffect(() => {
    const polkadot = new Polkadot();
    const crossChainInteroperabilityAPI = new CrossChainInteroperabilityAPI();

    polkadot.connect(user.id).then((connection) => {
      crossChainInteroperabilityAPI.initiateCrossChainTransaction(user.id).then((transaction) => {
        setCrossChainTransaction(transaction);
      });
    });
  }, [user]);

  return (
    <div>
      <h2>Cross-Chain Interoperability</h2>
      <p>Cross-Chain Transaction: {JSON.stringify(crossChainTransaction)}</p>
    </div>
  );
};

export default CrossChainInteroperability;
