import { handleActions } from 'redux-actions';
import { CREATE_RIDE_REQUEST, CREATE_RIDE_REQUEST_SUCCESS, CREATE_RIDE_REQUEST_FAILURE } from '../actions/rideActions';
import { GET_RIDE_REQUESTS, GET_RIDE_REQUESTS_SUCCESS, GET_RIDE_REQUESTS_FAILURE } from '../actions/rideActions';
import { CREATE_RIDE_OFFER, CREATE_RIDE_OFFER_SUCCESS, CREATE_RIDE_OFFER_FAILURE } from '../actions/rideActions';
import { GET_RIDE_OFFERS, GET_RIDE_OFFERS_SUCCESS, GET_RIDE_OFFERS_FAILURE } from '../actions/rideActions';

const initialState = {
  rideRequests: [],
  rideOffers: [],
  isLoading: false,
  error: null,
};

export default handleActions({
  [CREATE_RIDE_REQUEST]: (state, action) => ({
    ...state,
    isLoading: true,
  }),
  [CREATE_RIDE_REQUEST_SUCCESS]: (state, action) => ({
    ...state,
    isLoading: false,
    rideRequests: [...state.rideRequests, action.payload],
  }),
  [CREATE_RIDE_REQUEST_FAILURE]: (state, action) => ({
    ...state,
    isLoading: false,
    error: action.payload,
  }),
  [GET_RIDE_REQUESTS]: (state, action) => ({
    ...state,
    isLoading: true,
  }),
  [GET_RIDE_REQUESTS_SUCCESS]: (state, action) => ({
    ...state,
    isLoading: false,
    rideRequests: action.payload,
  }),
  [GET_RIDE_REQUESTS_FAILURE]: (state, action) => ({
    ...state,
    isLoading: false,
    error: action.payload,
  }),
  [CREATE_RIDE_OFFER]: (state, action) => ({
    ...state,
    isLoading: true,
  }),
  [CREATE_RIDE_OFFER_SUCCESS]: (state, action) => ({
    ...state,
    isLoading: false,
    rideOffers: [...state.rideOffers, action.payload],
  }),
  [CREATE_RIDE_OFFER_FAILURE]: (state, action) => ({
    ...state,
    isLoading: false,
    error: action.payload,
  }),
  [GET_RIDE_OFFERS]: (state, action) => ({
    ...state,
    isLoading: true,
  }),
  [GET_RIDE_OFFERS_SUCCESS]: (state, action) => ({
    ...state,
    isLoading: false,
    rideOffers: action.payload,
  }),
  [GET_RIDE_OFFERS_FAILURE]: (state, action) => ({
    ...state,
    isLoading: false,
    error: action.payload,
  }),
}, initialState);
