import React, { useState } from 'react';
import { useWeb3React } from '@web3-react/core';
import { AstralPlaneNFT } from '../contracts/AstralPlaneNFT';

const AstralPlaneNFTCreator = () => {
  const { account, library } = useWeb3React();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [price, setPrice] = useState(0);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const id = await fetch('/api/astralplane-nft', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description, price }),
    });
    console.log(`NFT created with ID ${id}`);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>Name:</label>
      <input type="text" value={name} onChange={(event) => setName(event.target.value)} />
      <br />
      <label>Description:</label>
      <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
      <br />
      <label>Price (ETH):</label>
      <input type="number" value={price} onChange={(event) => setPrice(event.target.value)} />
      <br />
      <button type="submit">Create NFT</button>
    </form>
  );
};

export default AstralPlaneNFTCreator;
