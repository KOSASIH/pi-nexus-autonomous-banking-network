import React, { useState, useEffect } from 'react';
import axios from 'axios';

const FundWallet = () => {
  const [walletAddress, setWalletAddress] = useState('');
  const [emailAddress, setEmailAddress] = useState('');
  const [redirectUrl, setRedirectUrl] = useState('');
  const [onrampUrl, setOnrampUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fundWallet = async () => {
      try {
        const onrampResponse = await axios.post('/api/onramp', {
          address: walletAddress,
          email: emailAddress,
          redirectUrl,
        });
        setOnrampUrl(onrampResponse.data.url);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };
    fundWallet();
  }, [walletAddress, emailAddress, redirectUrl]);

  const handleWalletAddressChange = (event) => {
    setWalletAddress(event.target.value);
  };

  const handleEmailAddressChange = (event) => {
    setEmailAddress(event.target.value);
  };

  const handleRedirectUrlChange = (event) => {
    setRedirectUrl(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setLoading(true);
  };

  return (
    <div>
      <h1>Fund Wallet</h1>
      <form onSubmit={handleSubmit}>
        <label>Wallet Address:</label>
        <input type="text" value={walletAddress} onChange={handleWalletAddressChange} />
        <br />
        <label>Email Address:</label>
        <input type="email" value={emailAddress} onChange={handleEmailAddressChange} />
        <br />
        <label>Redirect URL:</label>
        <input type="text" value={redirectUrl} onChange={handleRedirectUrlChange} />
        <br />
        <button type="submit">Fund Wallet</button>
      </form>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <p>
          {onrampUrl ? (
            <a href={onrampUrl} target="_blank" rel="noopener noreferrer">
              Click here to complete the on-ramp flow
            </a>
          ) : (
            <p>No on-ramp URL generated</p>
          )}
        </p>
      )}
      {error ? <p>Error: {error}</p> : null}
    </div>
  );
};

export default FundWallet;
