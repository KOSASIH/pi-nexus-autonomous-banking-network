import axios from 'axios';

export const fetchContracts = () => {
  return (dispatch) => {
    axios.get('/api/contracts')
      .then(response => {
        dispatch({
          type: 'FETCH_CONTRACTS_SUCCESS',
          contracts: response.data
        });
      })
      .catch(error => {
        dispatch({
          type: 'FETCH_CONTRACTS_FAILURE',
          error: error.message
        });
      });
  };
};

export const fetchContract = (id) => {
  return (dispatch) => {
    axios.get(`/api/contracts/${id}`)
      .then(response => {
        dispatch({
          type: 'FETCH_CONTRACT_SUCCESS',
          contract: response.data
        });
      })
      .catch(error => {
        dispatch({
          type: 'FETCH_CONTRACT_FAILURE',
          error: error.message
        });
      });
  };
};

export const createContract = (contract) => {
  return (dispatch) => {
    axios.post('/api/contracts', contract)
      .then(response => {
        dispatch({
          type: 'CREATE_CONTRACT_SUCCESS',
          contract: response.data
        });
      })
      .catch(error => {
        dispatch({
          type: 'CREATE_CONTRACT_FAILURE',
          error: error.message
        });
      });
  };
};
