// actions/walletActions.js
import { createAction } from 'redux-actions';
import { walletAPI } from '../api/walletAPI';

export const SET_WALLET_ADDRESS = 'SET_WALLET_ADDRESS';
export const SET_WALLET_BALANCE = 'SET_WALLET_BALANCE';
export const SET_WALLET_TRANSACTIONS = 'SET_WALLET_TRANSACTIONS';

export const setWalletAddress = createAction(SET_WALLET_ADDRESS, (address) => ({ address }));
export const setWalletBalance = createAction(SET_WALLET_BALANCE, (balance) => ({ balance }));
export const setWalletTransactions = createAction(SET_WALLET_TRANSACTIONS, (transactions) => ({ transactions }));

export const fetchWalletAddress = () => async (dispatch) => {
  try {
    const response = await walletAPI.getAddress();
    dispatch(setWalletAddress(response.data.address));
  } catch (error) {
    console.error(error);
  }
};

export const fetchWalletBalance = () => async (dispatch) => {
  try {
    const response = await walletAPI.getBalance();
    dispatch(setWalletBalance(response.data.balance));
  } catch (error) {
    console.error(error);
  }
};

export const fetchWalletTransactions = () => async (dispatch) => {
  try {
    const response = await walletAPI.getTransactions();
    dispatch(setWalletTransactions(response.data.transactions));
  } catch (error) {
    console.error(error);
  }
};

export const sendTransaction = (amount, recipient) => async (dispatch) => {
  try {
    const response = await walletAPI.sendTransaction(amount, recipient);
    dispatch(setWalletBalance(response.data.balance));
  } catch (error) {
    console.error(error);
  }
};
