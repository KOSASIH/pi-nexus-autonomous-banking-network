import { combineReducers } from 'redux';
import { authActions } from '../actions/auth.actions';
import { AuthState } from '../models/auth';

const initialState: AuthState = {
  accessToken: null,
  refreshToken: null,
  isAuthenticated: false,
};

const authReducer = (state = initialState, action: any) => {
  switch (action.type) {
    case authActions.LOGIN_SUCCESS:
      return { ...state, accessToken: action.accessToken, refreshToken: action.refreshToken, isAuthenticated: true };
    case authActions.LOGOUT_SUCCESS:
      return { ...state, accessToken: null, refreshToken: null, isAuthenticated: false };
    case authActions.REFRESH_TOKEN_SUCCESS:
      return { ...state, accessToken: action.accessToken, refreshToken: action.refreshToken };
    default:
      return state;
  }
};

const rootReducer = combineReducers({
  auth: authReducer,
});

export default rootReducer;
