import { combineReducers } from 'redux';

const transactionsReducer = (state = [], action) => {
  switch (action.type) {
    case 'GET_TRANSACTIONS_SUCCESS':
      return action.payload;
    default:
      return state;
  }
};

const transactionDetailsReducer = (state = {}, action) => {
  switch (action.type) {
    case 'GET_TRANSACTION_DETAILS_SUCCESS':
      return action.payload;
    default:
      return state;
  }
};

export default combineReducers({
  transactions: transactionsReducer,
  transactionDetails: transactionDetailsReducer,
});
