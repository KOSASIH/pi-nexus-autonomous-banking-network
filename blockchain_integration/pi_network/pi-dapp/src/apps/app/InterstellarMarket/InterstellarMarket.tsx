import React, { useState, useEffect } from 'eact';
import { useWeb3 } from '../utils/web3';
import { useIdentityVerification } from '../utils/identity-verification';
import { useMatchmaking } from '../utils/matchmaking';

const InterstellarMarket = () => {
  const [ listings, setListings ] = useState([]);
  const [ identity, setIdentity ] = useState(null);
  const web3 = useWeb3();
  const identityVerification = useIdentityVerification();
  const matchmaking = useMatchmaking();

  useEffect(() => {
    const fetchListings = async () => {
      const listings = await web3.eth.getStorageAt('0x...'); // Fetch listings from blockchain storage
      setListings(listings);
    };
    fetchListings();
  }, [web3]);

  useEffect(() => {
    const verifyIdentity = async () => {
      const identity = await identityVerification.verify(web3.eth.defaultAccount);
      setIdentity(identity);
    };
    verifyIdentity();
  }, [web3, identityVerification]);

  const handlePurchase = async (listingId: number) => {
    const listing = listings.find((listing) => listing.id === listingId);
    const buyer = identity;
    const seller = listing.seller;
    const transaction = await web3.eth.sendTransaction({
      from: buyer,
      to: seller,
      value: listing.price,
    });
    matchmaking.match(buyer, seller, listing); // Match buyer and seller using AI-powered matchmaking
  };

  return (
    <div>
      <h1>Interstellar Market</h1>
      <ul>
        {listings.map((listing) => (
          <li key={listing.id}>
            <p>{listing.name}</p>
<p>Price: {listing.price} ETH</p>
            <button onClick={() => handlePurchase(listing.id)}>Buy</button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default InterstellarMarket;
