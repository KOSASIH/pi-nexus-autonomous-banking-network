import { handleActions } from 'redux-actions';
import { UPDATE_USER_PROFILE, UPDATE_USER_PROFILE_SUCCESS, UPDATE_USER_PROFILE_FAILURE } from '../actions/userActions';
import { GET_USER_PROFILE, GET_USER_PROFILE_SUCCESS, GET_USER_PROFILE_FAILURE } from '../actions/userActions';

const initialState = {
  userProfile: null,
  isLoading: false,
  error: null,
};

export default handleActions({
  [UPDATE_USER_PROFILE]: (state, action) => ({
    ...state,
    isLoading: true,
  }),
  [UPDATE_USER_PROFILE_SUCCESS]: (state, action) => ({
    ...state,
    isLoading: false,
    userProfile: action.payload,
  }),
  [UPDATE_USER_PROFILE_FAILURE]: (state, action) => ({
    ...state,
    isLoading: false,
    error: action.payload,
  }),
  [GET_USER_PROFILE]: (state, action) => ({
    ...state,
    isLoading: true,
  }),
  [GET_USER_PROFILE_SUCCESS]: (state, action) => ({
    ...state,
    isLoading: false,
    userProfile: action.payload,
  }),
  [GET_USER_PROFILE_FAILURE]: (state, action) => ({
    ...state,
    isLoading: false,
    error: action.payload,
  }),
}, initialState);
