// reducers/walletReducer.js
import { combineReducers } from 'redux';
import { SET_WALLET_ADDRESS, SET_WALLET_BALANCE, SET_WALLET_TRANSACTIONS } from '../actions/walletActions';

const initialState = {
  address: '',
  balance: 0,
  transactions: [],
};

const walletReducer = (state = initialState, action) => {
  switch (action.type) {
    case SET_WALLET_ADDRESS:
      return { ...state, address: action.address };
    case SET_WALLET_BALANCE:
      return { ...state, balance: action.balance };
    case SET_WALLET_TRANSACTIONS:
      return { ...state, transactions: action.transactions };
    default:
      return state;
  }
};

export default walletReducer;
