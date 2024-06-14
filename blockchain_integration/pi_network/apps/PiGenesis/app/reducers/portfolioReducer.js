import { SET_USER_DATA, FETCH_PORTFOLIO_DATA } from '../actions';

const initialState = {
  userData: null,
  portfolioData: null,
};

const portfolioReducer = (state = initialState, action) => {
  switch (action.type) {
    case SET_USER_DATA:
      return {...state, userData: action.userData };
    case FETCH_PORTFOLIO_DATA:
      // Fetch portfolio data from API
      return {...state, portfolioData: action.portfolioData };
    default:
      return state;
  }
};

export default portfolioReducer;
