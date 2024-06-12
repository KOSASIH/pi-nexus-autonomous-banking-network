// components/Header.js
import React from 'eact';
import { Link } from 'eact-router-dom';
import { useSelector } from 'eact-redux';
import { selectWalletAddress } from '../reducers/walletReducer';

const Header = () => {
  const walletAddress = useSelector(selectWalletAddress);

  return (
    <header className="header">
      <div className="logo">
        <Link to="/">
          <img src="/logo.png" alt="Pi Browser" />
        </Link>
      </div>
      <nav>
        <ul>
          <li>
            <Link to="/browser">Browser</Link>
          </li>
          <li>
            <Link to="/wallet">Wallet</Link>
          </li>
          <li>
            <Link to="/settings">Settings</Link>
          </li>
        </ul>
      </nav>
      <div className="wallet-info">
        <span>Wallet Address:</span>
        <span>{walletAddress}</span>
      </div>
    </header>
  );
};

export default Header;
