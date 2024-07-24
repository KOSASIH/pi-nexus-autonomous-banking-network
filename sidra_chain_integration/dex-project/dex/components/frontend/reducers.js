import { combineReducers } from 'redux';

const balanceReducer = (state = 0, action) => {
  switch (action.type) {
    case 'FETCH_BALANCE_SUCCESS':
      return action.balance;
    default:
      return state;
  }
};

const transactionHistoryReducer = (state = [], action) => {
  switch (action.type) {
    case 'FETCH_TRANSACTION_HISTORY_SUCCESS':
      return action.transactionHistory;
    default:
      return state;
  }
};

export default combineReducers({
  balance: balanceReducer,
  transactionHistory: transactionHistoryReducer,
});
