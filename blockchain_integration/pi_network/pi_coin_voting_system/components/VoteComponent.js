import React, { useState, useEffect } from 'react';
import { connect } from 'react-redux';
import { castVote, getVotes, resetVote } from '../actions/vote.actions';

const VoteComponent = ({ castVote, getVotes, resetVote, voteCount, voteAverage, voteStandardDeviation, isVoting, error }) => {
  const [voteValue, setVoteValue] = useState(0);

  useEffect(() => {
    getVotes();
  }, []);

  const handleVoteChange = (e) => {
    setVoteValue(e.target.value);
  };

  const handleCastVote = () => {
    castVote(voteValue);
  };

  const handleResetVote = () => {
    resetVote();
  };

  return (
    <div>
      <h1>Vote Component</h1>
      <form>
        <label>Vote Value:</label>
        <input type="number" value={voteValue} onChange={handleVoteChange} />
        <button onClick={handleCastVote}>Cast Vote</button>
        <button onClick={handleResetVote}>Reset Vote</button>
      </form>
      {isVoting ? (
        <p>Voting...</p>
      ) : (
        <div>
          <p>Vote Count: {voteCount}</p>
          <p>Vote Average: {voteAverage}</p>
          <p>Vote Standard Deviation: {voteStandardDeviation}</p>
          {error ? <p>Error: {error}</p> : null}
        </div>
      )}
    </div>
  );
};

const mapStateToProps = (state) => {
  return {
    voteCount: state.vote.voteCount,
    voteAverage: state.vote.voteAverage,
    voteStandardDeviation: state.vote.voteStandardDeviation,
    isVoting: state.vote.isVoting,
    error: state.vote.error
  };
};

export default connect(mapStateToProps, { castVote, getVotes, resetVote })(VoteComponent);
