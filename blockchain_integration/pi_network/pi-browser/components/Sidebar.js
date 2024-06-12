// components/Sidebar.js
import React from 'eact';
import { Link } from 'eact-router-dom';

const Sidebar = () => {
  return (
    <aside className="sidebar">
      <ul>
        <li>
          <Link to="/browser">
            <i className="fas fa-globe" />
            Browser
          </Link>
        </li>
        <li>
          <Link to="/wallet">
            <i className="fas fa-wallet" />
            Wallet
          </Link>
        </li>
        <li>
          <Link to="/settings">
            <i className="fas fa-cog" />
            Settings
          </Link>
        </li>
        <li>
          <Link to="/about">
            <i className="fas fa-info" />
            About
          </Link>
        </li>
      </ul>
    </aside>
  );
};

export default Sidebar;
