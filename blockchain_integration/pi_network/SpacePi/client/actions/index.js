export const fetchLaunches = () => async dispatch => {
  try {
    const response = await axios.get('/api/launches');
    dispatch({ type: 'FETCH_LAUNCHES_SUCCESS', launches: response.data });
  } catch (error) {
    dispatch({ type: 'FETCH_LAUNCHES_FAILURE', error });
  }
};

export const fetchMerchandise = () => async dispatch => {
  try {
    const response = await axios.get('/api/merchandise');
    dispatch({ type: 'FETCH_MERCHANDISE_SUCCESS', merchandise: response.data });
  } catch (error) {
    dispatch({ type: 'FETCH_MERCHANDISE_FAILURE', error });
  }
};

export const fetchUsers = () => async dispatch => {
  try {
    const response = await axios.get('/api/users');
    dispatch({ type: 'FETCH_USERS_SUCCESS', users: response.data });
  } catch (error) {
    dispatch({ type: 'FETCH_USERS_FAILURE', error });
  }
};
