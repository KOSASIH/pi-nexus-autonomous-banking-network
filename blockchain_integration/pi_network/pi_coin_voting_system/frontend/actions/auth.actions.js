export const login = (username, password) => async dispatch => {
  try {
    const response = await axios.post('/api/login', { username, password });
    const token = response.data.token;
    dispatch({ type: 'LOGIN_SUCCESS', token });
  } catch (err) {
    dispatch({ type: 'LOGIN_FAILURE', error: err.response.data.error });
  }
};

export const register = (username, password, email) => async dispatch => {
  try {
    const response = await axios.post('/api/register', { username, password, email });
    dispatch({ type: 'REGISTER_SUCCESS' });
  } catch (err) {
    dispatch({ type: 'REGISTER_FAILURE', error: err.response.data.error });
  }
};
