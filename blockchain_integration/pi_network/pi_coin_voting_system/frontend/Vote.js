import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Vote = () => {
  const [vote, setVote] = useState(0);
  const [error, setError] = useState(null);
  const [voteCount, setVoteCount] = useState(0);
  const [voteAverage, setVoteAverage] = useState(0);
  const [voteStandardDeviation, setVoteStandardDeviation] = useState(0);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const headers = { Authorization: `Bearer ${token}` };
    axios
      .get('/api/getVotes', { headers })
      .then((response) => {
        const { voteCount, voteAverage, voteStandardDeviation } = response.data;
        setVoteCount(voteCount);
        setVoteAverage(voteAverage);
        setVoteStandardDeviation(voteStandardDeviation);
      })
      .catch((err) => {
        setError(err.response.data.error);
      });
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      const response = await axios.post('/api/castVote', { vote }, { headers });
      console.log(response.data);
      setVote(0);
    } catch (err) {
      setError(err.response.data.error);
    }
  };

  return (
    <div>
      <h1>Cast Your Vote!</h1>
      <form onSubmit={handleSubmit}>
        <label>Enter your vote:</label>
        <input
          type="number"
          value={vote}
          onChange={(e) => setVote(e.target.value)}
        />
        <br />
        <button type="submit">Cast Vote</button>
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <h2>Vote Statistics:</h2>
      <p>Vote Count: {voteCount}</p>
      <p>Average Vote: {voteAverage.toFixed(2)}</p>
      <p>Standard Deviation: {voteStandardDeviation.toFixed(2)}</p>
    </div>
  );
};

export default Vote;
