import React from 'react';
import { Link } from 'react-router-dom';

const Nav = () => {
  return (
    <nav>
      <ul>
        <li>
          <Link to="/">Dashboard</Link>
        </li>
        <li>
          <Link to="/medical-billings">Medical Billings</Link>
        </li>
        <li>
          <Link to="/patients">Patients</Link>
        </li>
        <li>
          <Link to="/healthcare-providers">Healthcare Providers</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Nav;
