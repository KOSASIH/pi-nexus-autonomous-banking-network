import React, { useState, useEffect } from 'react';
import { Polkadot } from 'polkadot-js';
import { CrossChainAtomicSwapsAPI } from '../api/cross-chain-atomic-swaps';

interface CrossChainAtomicSwapsProps {
  user: any;
}

const CrossChainAtomicSwaps: React.FC<CrossChainAtomicSwapsProps> = ({ user }) => {
  const [atomicSwap, setAtomicSwap] = useState({});

  useEffect(() => {
    const polkadot = new Polkadot();
    const crossChainAtomicSwapsAPI = new CrossChainAtomicSwapsAPI();

    polkadot.connect(user.id).then((connection) => {
      crossChainAtomicSwapsAPI.initiateAtomicSwap(user.id).then((swap) => {
        setAtomicSwap(swap);
      });
    });
  }, [user]);

  return (
    <div>
      <h2>Cross-Chain Atomic Swaps</h2>
      <p>Atomic Swap: {JSON.stringify(atomicSwap)}</p>
    </div>
  );
};

export default CrossChainAtomicSwaps;
