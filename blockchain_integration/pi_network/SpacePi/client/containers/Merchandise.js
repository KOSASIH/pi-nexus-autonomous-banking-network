import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MerchandiseCard from '../components/MerchandiseCard';

const Merchandise = () => {
  const [merchandise, setMerchandise] = useState([]);

  useEffect(() => {
    axios.get('/api/merchandise')
      .then(response => {
        setMerchandise(response.data);
      })
      .catch(error => {
        console.error(error);
      });
   }, []);

  return (
    <div className="merchandise">
      <h1>Merchandise</h1>
      <ul>
        {merchandise.map(item => (
          <li key={item.id}>
            <MerchandiseCard merchandise={item} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Merchandise;
