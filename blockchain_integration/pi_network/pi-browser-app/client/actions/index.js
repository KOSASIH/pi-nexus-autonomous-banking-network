import axios from 'axios';

export const getTransactions = () => async (dispatch) => {
  try {
    const response = await axios.get('/api/transactions');
    dispatch({ type: 'GET_TRANSACTIONS_SUCCESS', payload: response.data });
  } catch (error) {
    dispatch({ type: 'GET_TRANSACTIONS_FAILURE', payload: error.message });
  }
};

export const getTransactionDetails = (id) => async (dispatch) => {
  try {
    const response = await axios.get(`/api/transactions/${id}`);
    dispatch({ type: 'GET_TRANSACTION_DETAILS_SUCCESS', payload: response.data });
  } catch (error) {
    dispatch({ type: 'GET_TRANSACTION_DETAILS_FAILURE', payload: error.message });
  }
};
