import React from 'react';
import { Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSpace Shuttle } from '@fortawesome/free-solid-svg-icons';

const Header = () => {
  return (
    <header className="header">
      <nav className="nav">
        <ul>
          <li>
            <Link to="/" className="nav-link">
              <FontAwesomeIcon icon={faSpaceShuttle} /> SpacePi
            </Link>
          </li>
          <li>
            <Link to="/launches" className="nav-link">
              Launches
            </Link>
          </li>
          <li>
            <Link to="/merchandise" className="nav-link">
              Merchandise
            </Link>
          </li>
          <li>
            <Link to="/users" className="nav-link">
              Users
            </Link>
          </li>
        </ul>
      </nav>
    </header>
  );
};

export default Header;
