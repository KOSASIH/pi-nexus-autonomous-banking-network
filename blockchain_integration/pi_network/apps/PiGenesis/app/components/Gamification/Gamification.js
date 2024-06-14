import React, { useState, useEffect } from 'eact';
import axios from 'axios';

const Gamification = () => {
  const [rewards, setRewards] = useState([]);
  const [points, setPoints] = useState(0);

  useEffect(() => {
    axios.get('/api/rewards')
     .then((response) => {
        setRewards(response.data);
      })
     .catch((error) => {
        console.error(error);
      });

    axios.get('/api/points')
     .then((response) => {
        setPoints(response.data);
      })
     .catch((error) => {
        console.error(error);
      });
  }, []);

  const handleRedeemReward = (rewardId) => {
    axios.post('/api/redeem-reward', { rewardId })
     .then((response) => {
        setPoints(response.data);
      })
     .catch((error) => {
        console.error(error);
      });
  };

  return (
    <div>
      <h2>Gamification</h2>
      <p>Earn points and redeem rewards:</p>
      <ul>
        {rewards.map((reward) => (
          <li key={reward.id}>
            {reward.name}: {reward.pointsRequired} points
            <button onClick={() => handleRedeemReward(reward.id)}>Redeem</button>
          </li>
        ))}
      </ul>
      <p>Current Points: {points}</p>
    </div>
  );
};

export default Gamification;
