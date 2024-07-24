const initialState = {
  voteCount: 0,
  voteAverage: 0,
  voteStandardDeviation: 0,
  isVoting: false,
  error: null
};

export default function voteReducer(state = initialState, action) {
  switch (action.type) {
    case 'CAST_VOTE_SUCCESS':
      return { ...state, isVoting: false };
    case 'CAST_VOTE_FAILURE':
      return { ...state, error: action.error, isVoting: false };
    case 'GET_VOTES_SUCCESS':
      return { ...state, voteCount: action.voteCount, voteAverage: action.voteAverage, voteStandardDeviation: action.voteStandardDeviation };
    case 'GET_VOTES_FAILURE':
      return { ...state, error: action.error };
    case 'RESET_VOTE':
      return initialState;
    default:
      return state;
  }
}
