import React, { useState, useEffect } from 'react';
import axios from 'axios';
import LaunchCard from '../components/LaunchCard';

const Launches = () => {
  const [launches, setLaunches] = useState([]);

  useEffect(() => {
    axios.get('/api/launches')
      .then(response => {
        setLaunches(response.data);
      })
      .catch(error => {
        console.error(error);
      });
  }, []);

  return (
    <div className="launches">
      <h1>Launches</h1>
      <ul>
        {launches.map(launch => (
          <li key={launch.id}>
            <LaunchCard launch={launch} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Launches;
