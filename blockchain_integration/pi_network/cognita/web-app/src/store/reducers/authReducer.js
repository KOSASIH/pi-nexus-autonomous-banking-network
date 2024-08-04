import { AUTH_LOGIN, AUTH_LOGOUT, AUTH_ERROR } from '../actions/authActions';

const initialState = {
  token: null,
  error: null
};

export default function authReducer(state = initialState, action) {
  switch (action.type) {
    case AUTH_LOGIN:
      return {...state, token: action.token };
    case AUTH_LOGOUT:
      return {...state, token: null };
    case AUTH_ERROR:
      return {...state, error: action.error };
    default:
      return state;
  }
}
