import { createAction, createReducer, createSaga } from 'edux-act';
import { takeEvery, put, call, select } from 'edux-saga/effects';
import { getBalance, getTransactionHistory } from '../api';
import { toast } from 'eact-toastify';

// Actions
export const FETCH_BALANCE = 'FETCH_BALANCE';
export const FETCH_BALANCE_SUCCESS = 'FETCH_BALANCE_SUCCESS';
export const FETCH_BALANCE_FAILURE = 'FETCH_BALANCE_FAILURE';

export const FETCH_TRANSACTION_HISTORY = 'FETCH_TRANSACTION_HISTORY';
export const FETCH_TRANSACTION_HISTORY_SUCCESS = 'FETCH_TRANSACTION_HISTORY_SUCCESS';
export const FETCH_TRANSACTION_HISTORY_FAILURE = 'FETCH_TRANSACTION_HISTORY_FAILURE';

export const fetchBalance = createAction(FETCH_BALANCE, (address) => ({ address }));
export const fetchTransactionHistory = createAction(FETCH_TRANSACTION_HISTORY, (address) => ({ address }));

// Reducer
const initialState = {
  balance: 0,
  transactionHistory: [],
  loading: false,
  error: null,
};

const reducer = createReducer(initialState, {
  [FETCH_BALANCE_SUCCESS]: (state, { balance }) => ({...state, balance, loading: false }),
  [FETCH_BALANCE_FAILURE]: (state, { error }) => ({...state, error, loading: false }),
  [FETCH_TRANSACTION_HISTORY_SUCCESS]: (state, { transactionHistory }) => ({...state, transactionHistory, loading: false }),
  [FETCH_TRANSACTION_HISTORY_FAILURE]: (state, { error }) => ({...state, error, loading: false }),
});

export default reducer;

// Saga
function* fetchBalanceSaga({ address }) {
  try {
    const balance = yield call(getBalance, address);
    yield put({ type: FETCH_BALANCE_SUCCESS, balance });
  } catch (error) {
    yield put({ type: FETCH_BALANCE_FAILURE, error });
    toast.error(`Error fetching balance: ${error.message}`);
  }
}

function* fetchTransactionHistorySaga({ address }) {
  try {
    const transactionHistory = yield call(getTransactionHistory, address);
    yield put({ type: FETCH_TRANSACTION_HISTORY_SUCCESS, transactionHistory });
  } catch (error) {
    yield put({ type: FETCH_TRANSACTION_HISTORY_FAILURE, error });
    toast.error(`Error fetching transaction history: ${error.message}`);
  }
}

export function* rootSaga() {
  yield takeEvery(FETCH_BALANCE, fetchBalanceSaga);
  yield takeEvery(FETCH_TRANSACTION_HISTORY, fetchTransactionHistorySaga);
}
