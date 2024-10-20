import React from 'react';

const Badge = ({ badge }) => {
  return (
    <div>
      <h2>{badge.title}</h2>
      <p>{badge.description}</p>
    </div>
  );
};

export default Badge;
