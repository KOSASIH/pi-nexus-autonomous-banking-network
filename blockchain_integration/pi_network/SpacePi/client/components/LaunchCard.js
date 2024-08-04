import React from 'react';
import { Link } from 'react-router-dom';

const LaunchCard = ({ launch }) => {
  return (
    <div className="launch-card">
      <h2>{launch.name}</h2>
      <p>{launch.description}</p>
      <p>
        <Link to={`/launches/${launch.id}`}>View Details</Link>
      </p>
    </div>
  );
};

export default LaunchCard;
