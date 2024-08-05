import React from 'react';
import { Link } from 'react-router-dom';

const MerchandiseCard = ({ merchandise }) => {
  return (
    <div className="merchandise-card">
      <h2>{merchandise.name}</h2>
      <p>{merchandise.description}</p>
      <p>
        <Link to={`/merchandise/${merchandise.id}`}>View Details</Link>
      </p>
    </div>
  );
};

export default MerchandiseCard;
