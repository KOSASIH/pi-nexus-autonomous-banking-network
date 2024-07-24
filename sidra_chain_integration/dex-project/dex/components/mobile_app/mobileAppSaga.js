import { takeEvery, put, call } from 'redux-saga/effects';
import { fetchBalance, fetchTransactionHistory } from '../api';

function* fetchBalanceSaga() {
  try {
    const balance = yield call(fetchBalance);
    yield put({ type: FETCH_BALANCE_SUCCESS, balance });
  } catch (error) {
    yield put({ type: FETCH_BALANCE_FAILURE, error });
  }
}

function* fetchTransactionHistorySaga() {
  try {
    const transactionHistory = yield call(fetchTransactionHistory);
    yield put({ type: FETCH_TRANSACTION_HISTORY_SUCCESS, transactionHistory });
  } catch (error) {
    yield put({ type: FETCH_TRANSACTION_HISTORY_FAILURE, error });
  }
}

export function* rootSaga() {
  yield takeEvery(FETCH_BALANCE, fetchBalanceSaga);
  yield takeEvery(FETCH_TRANSACTION_HISTORY, fetchTransactionHistorySaga);
}
