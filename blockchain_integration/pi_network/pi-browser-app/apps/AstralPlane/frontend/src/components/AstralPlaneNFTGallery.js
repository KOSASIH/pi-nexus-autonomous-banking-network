import React, { useState, useEffect } from 'react';
import { useWeb3React } from '@web3-react/core';
import { AstralPlaneNFT } from '../contracts/AstralPlaneNFT';

const AstralPlaneNFTGallery = () => {
  const { account, library } = useWeb3React();
  const [nfts, setNfts] = useState([]);
  const [ownedNfts, setOwnedNfts] = useState([]);

  useEffect(() => {
    const fetchNfts = async () => {
      const nfts = await AstralPlaneNFT.getNfts();
      setNfts(nfts);
    };
    fetchNfts();
  }, [AstralPlaneNFT]);

  useEffect(() => {
    const fetchOwnedNfts = async () => {
      const ownedNfts = await AstralPlaneNFT.getOwnedNfts(account);
      setOwnedNfts(ownedNfts);
    };
    fetchOwnedNfts();
  }, [account, AstralPlaneNFT]);

  const handleBuyNft = async (id) => {
    await AstralPlaneNFT.transferNft(id, account);
  };

  return (
    <div>
      <h1>AstralPlane NFT Gallery</h1>
      <ul>
        {nfts.map((nft, index) => (
          <li key={index}>
            <img src={nft.image} alt={nft.name} />
            <p>{nft.name}</p>
            <p>{nft.description}</p>
            <p>Price: {nft.price} ETH</p>
            {ownedNfts.includes(nft.id) ? (
              <p>You own this NFT!</p>
            ) : (
              <button onClick={() => handleBuyNft(nft.id)}>Buy NFT</button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AstralPlaneNFTGallery;
