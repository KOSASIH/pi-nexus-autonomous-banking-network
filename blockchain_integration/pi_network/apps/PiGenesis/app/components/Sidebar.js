import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useWeb3React } from '@web3-react/core';
import { Web3Provider } from '@ethersproject/providers';
import { useEthers } from '@ethersproject/ethers-react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDashboard, faPortfolio, faSettings } from '@fortawesome/free-solid-svg-icons';

const Sidebar = () => {
  const [menuItems, setMenuItems] = useState([
    {
      label: 'Dashboard',
      icon: faDashboard,
      path: '/',
    },
    {
      label: 'Portfolio',
      icon: faPortfolio,
      path: '/portfolio',
    },
    {
      label: 'Settings',
      icon: faSettings,
      path: '/settings',
    },
  ]);

  const { account, library } = useWeb3React();
  const { ethers } = useEthers();

  useEffect(() => {
    const fetchMenuItems = async () => {
      try {
        const response = await fetch('https://api.pi-genesis.com/menu-items');
        const data = await response.json();
        setMenuItems(data);
      } catch (error) {
        console.error(error);
      }
    };
   fetchMenuItems();
  }, []);

  return (
    <aside>
      <div className="sidebar">
        <ul>
          {menuItems.map((menuItem, index) => (
            <li key={index}>
              <Link to={menuItem.path}>
                <FontAwesomeIcon icon={menuItem.icon} size="lg" />
                <span>{menuItem.label}</span>
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
};

export default Sidebar;
