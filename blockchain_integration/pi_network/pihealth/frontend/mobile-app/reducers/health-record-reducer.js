import { combineReducers } from 'redux';
import {
  GET_HEALTH_RECORDS_REQUEST,
  GET_HEALTH_RECORDS_SUCCESS,
  GET_HEALTH_RECORDS_FAILURE,
  CREATE_HEALTH_RECORD_REQUEST,
  CREATE_HEALTH_RECORD_SUCCESS,
  CREATE_HEALTH_RECORD_FAILURE,
  UPDATE_HEALTH_RECORD_REQUEST,
  UPDATE_HEALTH_RECORD_SUCCESS,
  UPDATE_HEALTH_RECORD_FAILURE,
  DELETE_HEALTH_RECORD_REQUEST,
  DELETE_HEALTH_RECORD_SUCCESS,
  DELETE_HEALTH_RECORD_FAILURE,
} from '../actions/health-record-actions';

const initialState = {
  healthRecords: [],
  isLoading: false,
  error: null,
};

const healthRecordsReducer = (state = initialState, action) => {
  switch (action.type) {
    case GET_HEALTH_RECORDS_REQUEST:
      return { ...state, isLoading: true };
    case GET_HEALTH_RECORDS_SUCCESS:
      return { ...state, healthRecords: action.payload, isLoading: false };
    case GET_HEALTH_RECORDS_FAILURE:
      return { ...state, error: action.payload, isLoading: false };
    case CREATE_HEALTH_RECORD_REQUEST:
      return { ...state, isLoading: true };
    case CREATE_HEALTH_RECORD_SUCCESS:
      return { ...state, healthRecords: [...state.healthRecords, action.payload], isLoading: false };
    case CREATE_HEALTH_RECORD_FAILURE:
      return { ...state, error: action.payload, isLoading: false };
    case UPDATE_HEALTH_RECORD_REQUEST:
      return { ...state, isLoading: true };
    case UPDATE_HEALTH_RECORD_SUCCESS:
      return {
        ...state,
        healthRecords: state.healthRecords.map((healthRecord) =>
          healthRecord.id === action.payload.id ? action.payload : healthRecord
        ),
        isLoading: false,
      };
    case UPDATE_HEALTH_RECORD_FAILURE:
      return { ...state, error: action.payload, isLoading: false };
    case DELETE_HEALTH_RECORD_REQUEST:
      return { ...state, isLoading: true };
    case DELETE_HEALTH_RECORD_SUCCESS:
      return {
        ...state,
        healthRecords: state.healthRecords.filter((healthRecord) => healthRecord.id !== action.payload),
        isLoading: false,
      };
    case DELETE_HEALTH_RECORD_FAILURE:
      return { ...state, error: action.payload, isLoading: false };
    default:
      return state;
  }
};

export default combineReducers({ healthRecords: healthRecordsReducer });
