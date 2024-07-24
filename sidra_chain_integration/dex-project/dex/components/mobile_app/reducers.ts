import { createReducer } from '@reduxjs/toolkit';
import { fetchBalance, fetchTransactionHistory, sendTransaction } from './actions';

const initialState = {
  balance: 0,
  transactionHistory: [],
  loading: false,
};

const mobileAppReducer = createReducer(initialState, {
  [fetchBalance.success]: (state, action) => {
    state.balance = action.payload;
  },
  [fetchTransactionHistory.success]: (state, action) => {
    state.transactionHistory = action.payload;
  },
  [sendTransaction.pending]: (state) => {
    state.loading = true;
  },
  [sendTransaction.fulfilled]: (state) => {
    state.loading = false;
  },
  [sendTransaction.rejected]: (state) => {
    state.loading = false;
  },
});

export default mobileAppReducer;
