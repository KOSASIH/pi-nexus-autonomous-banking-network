import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useWeb3React } from '@web3-react/core';
import { Web3Provider } from '@ethersproject/providers';
import { useEthers } from '@ethersproject/ethers-react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTwitter, faDiscord, faGithub } from '@fortawesome/free-brands-svg-icons';

const Footer = () => {
  const [socialMediaLinks, setSocialMediaLinks] = useState({
    twitter: 'https://twitter.com/PiGenesis',
    discord: 'https://discord.gg/pi-genesis',
    github: 'https://github.com/PiGenesis/pi-genesis',
  });

  const { account, library } = useWeb3React();
  const { ethers } = useEthers();

  useEffect(() => {
    const fetchSocialMediaLinks = async () => {
      try {
        const response = await fetch('https://api.pi-genesis.com/social-media-links');
        const data = await response.json();
        setSocialMediaLinks(data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchSocialMediaLinks();
  }, []);

  const handleConnectWallet = async () => {
    try {
      await library.send('eth_requestAccounts', []);
      const accounts = await library.send('eth_accounts', []);
      setAccount(accounts[0]);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <footer>
      <div className="container">
        <div className="row">
          <div className="col-md-4">
            <h5>About PiGenesis</h5>
            <p>
              PiGenesis is a decentralized finance (DeFi) platform built on the Pi Network.
              Our mission is to empower individuals to take control of their financial lives.
            </p>
          </div>
          <div className="col-md-4">
            <h5>Resources</h5>
            <ul>
              <li>
                <Link to="/docs">Documentation</Link>
              </li>
              <li>
                <Link to="/faq">FAQ</Link>
              </li>
              <li>
                <Link to="/community">Community</Link>
              </li>
            </ul>
          </div>
          <div className="col-md-4">
            <h5>Connect with us</h5>
            <ul>
              <li>
                <a href={socialMediaLinks.twitter} target="_blank" rel="noopener noreferrer">
                  <FontAwesomeIcon icon={faTwitter} size="lg" />
                </a>
              </li>
              <li>
                <a href={socialMediaLinks.discord} target="_blank" rel="noopener noreferrer">
                  <FontAwesomeIcon icon={faDiscord} size="lg" />
                </a>
              </li>
              <li>
                <a href={socialMediaLinks.github} target="_blank" rel="noopener noreferrer">
                  <FontAwesomeIcon icon={faGithub} size="lg" />
                </a>
              </li>
            </ul>
            {account ? (
              <p>
                Connected to {account}
              </p>
            ) : (
              <button onClick={handleConnectWallet}>Connect Wallet</button>
            )}
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
