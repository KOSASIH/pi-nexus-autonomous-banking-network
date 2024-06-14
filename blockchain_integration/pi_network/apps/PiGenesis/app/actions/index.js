import { createAction } from 'edux-actions';

export const SET_USER_DATA = 'SET_USER_DATA';
export const setUserData = createAction(SET_USER_DATA, (userData) => ({ userData }));

export const FETCH_PORTFOLIO_DATA = 'FETCH_PORTFOLIO_DATA';
export const fetchPortfolioData = createAction(FETCH_PORTFOLIO_DATA, () => ({}));
