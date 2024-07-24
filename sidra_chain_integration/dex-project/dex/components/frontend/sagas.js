import { takeEvery, put, call } from 'redux-saga/effects';
import { fetchBalance, fetchTransactionHistory } from '../actions';
import { getBalance, getTransactionHistory } from '../api';

function* fetchBalanceSaga(action) {
  try {
    const balance = yield call(getBalance, action.address);
    yield put({ type: 'FETCH_BALANCE_SUCCESS', balance });
  } catch (error) {
    yield put({ type: 'FETCH_BALANCE_FAILURE', error });
  }
}

function* fetchTransactionHistorySaga(action) {
  try {
    const transactionHistory = yield call(getTransactionHistory, action.address);
    yield put({ type: 'FETCH_TRANSACTION_HISTORY_SUCCESS', transactionHistory });
  } catch (error) {
    yield put({ type: 'FETCH_TRANSACTION_HISTORY_FAILURE', error });
  }
}

export default function* rootSaga() {
  yield takeEvery('FETCH_BALANCE', fetchBalanceSaga);
  yield takeEvery('FETCH_TRANSACTION_HISTORY', fetchTransactionHistorySaga);
}
