import React, { useState, useEffect } from 'eact';
import axios from 'axios';

const SocialImpact = () => {
  const [impactData, setImpactData] = useState(null);

  useEffect(() => {
    axios.get('/api/social-impact')
     .then((response) => {
        setImpactData(response.data);
      })
     .catch((error) => {
        console.error(error);
      });
  }, []);

  return (
    <div>
      <h2>Social Impact</h2>
      <p>Our users have made a positive impact on the environment and society:</p>
      <ul>
        {impactData.map((impact) => (
          <li key={impact.id}>
            {impact.description}: {impact.value} {impact.unit}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default SocialImpact;
