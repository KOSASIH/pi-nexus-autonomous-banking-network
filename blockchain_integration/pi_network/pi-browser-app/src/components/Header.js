import React from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars, faTimes } from '@fortawesome/free-solid-svg-icons';

const Header = () => {
  const [menuOpen, setMenuOpen] = React.useState(false);

  const handleMenuToggle = () => {
    setMenuOpen(!menuOpen);
  };

  return (
    <header className="header">
      <nav className="nav">
        <ul className="nav-list">
          <li className="nav-item">
            <Link to="/" className="nav-link">
              Home
            </Link>
          </li>
          <li className="nav-item">
            <Link to="/blockchain" className="nav-link">
              Blockchain
            </Link>
          </li>
          <li className="nav-item">
            <Link to="/neural-network" className="nav-link">
              Neural Network
            </Link>
          </li>
          <li className="nav-item">
            <Link to="/artificial-intelligence" className="nav-link">
              Artificial Intelligence
            </Link>
          </li>
        </ul>
        <div className="nav-toggle" onClick={handleMenuToggle}>
          {menuOpen ? (
            <FontAwesomeIcon icon={faTimes} size="lg" />
          ) : (
            <FontAwesomeIcon icon={faBars} size="lg" />
          )}
        </div>
      </nav>
    </header>
  );
};

export default Header;
