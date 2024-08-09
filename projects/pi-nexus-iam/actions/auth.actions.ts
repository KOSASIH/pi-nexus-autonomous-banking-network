import { createAction, createReducer } from '@reduxjs/toolkit';
import { AuthState } from '../models/auth';
import { login, logout, refreshToken } from '../services/auth.service';

export const loginAction = createAction('auth/login', (email: string, password: string) => {
  return async (dispatch: any) => {
    try {
      const response = await login(email, password);
      dispatch(setAuthState(response.data));
    } catch (error) {
      console.error(error);
    }
  };
});

export const logoutAction = createAction('auth/logout', () => {
  return async (dispatch: any) => {
    try {
      await logout();
      dispatch(setAuthState(null));
    } catch (error) {
      console.error(error);
    }
  };
});

export const refreshTokenAction = createAction('auth/refreshToken', () => {
  return async (dispatch: any) => {
    try {
      const response = await refreshToken();
      dispatch(setAuthState(response.data));
    } catch (error) {
      console.error(error);
    }
  };
});

const setAuthState = createAction('auth/setAuthState', (authState: AuthState) => {
  return { authState };
});

const initialState: AuthState = null;

const authReducer = createReducer(initialState, {
  [setAuthState]: (state, action) => action.authState,
});

export default authReducer;
