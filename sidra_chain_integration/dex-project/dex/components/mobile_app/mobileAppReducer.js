import { FETCH_BALANCE_SUCCESS, FETCH_TRANSACTION_HISTORY_SUCCESS } from '../actions';

const initialState = {
  balance: 0,
  transactionHistory: [],
  loading: false,
};

export default function reducer(state = initialState, action) {
  switch (action.type) {
    case FETCH_BALANCE_SUCCESS:
      return { ...state, balance: action.balance, loading: false };
    case FETCH_TRANSACTION_HISTORY_SUCCESS:
      return { ...state, transactionHistory: action.transactionHistory, loading: false };
    default:
      return state;
  }
}
