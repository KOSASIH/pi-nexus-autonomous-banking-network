import { createAction } from '@reduxjs/toolkit';

export const fetchBalance = createAction('FETCH_BALANCE');
export const fetchTransactionHistory = createAction('FETCH_TRANSACTION_HISTORY');
export const sendTransaction = createAction('SEND_TRANSACTION');
