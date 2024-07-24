export const castVote = (vote) => async dispatch => {
  try {
    const token = localStorage.getItem('token');
    const headers = { Authorization: `Bearer ${token}` };
    const response = await axios.post('/api/castVote', { vote }, { headers });
    dispatch({ type: 'CAST_VOTE_SUCCESS' });
  } catch (err) {
    dispatch({ type: 'CAST_VOTE_FAILURE', error: err.response.data.error });
  }
};

export const getVotes = () => async dispatch => {
  try {
    const token = localStorage.getItem('token');
    const headers = { Authorization: `Bearer ${token}` };
    const response = await axios.get('/api/getVotes', { headers });
    const { voteCount, voteAverage, voteStandardDeviation } = response.data;
    dispatch({ type: 'GET_VOTES_SUCCESS', voteCount, voteAverage, voteStandardDeviation });
  } catch (err) {
    dispatch({ type: 'GET_VOTES_FAILURE', error: err.response.data.error });
  }
};

export const resetVote = () => async dispatch => {
  dispatch({ type: 'RESET_VOTE' });
};
