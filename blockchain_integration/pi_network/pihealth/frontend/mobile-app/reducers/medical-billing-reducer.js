import { combineReducers } from 'redux';
import {
  GET_MEDICAL_BILLS_REQUEST,
  GET_MEDICAL_BILLS_SUCCESS,
  GET_MEDICAL_BILLS_FAILURE,
  CREATE_MEDICAL_BILL_REQUEST,
  CREATE_MEDICAL_BILL_SUCCESS,
  CREATE_MEDICAL_BILL_FAILURE,
  UPDATE_MEDICAL_BILL_REQUEST,
  UPDATE_MEDICAL_BILL_SUCCESS,
  UPDATE_MEDICAL_BILL_FAILURE,
  DELETE_MEDICAL_BILL_REQUEST,
  DELETE_MEDICAL_BILL_SUCCESS,
  DELETE_MEDICAL_BILL_FAILURE,
} from '../actions/medical-billing-actions';

const initialState = {
  medicalBills: [],
  isLoading: false,
  error: null,
};

const medicalBillsReducer = (state = initialState, action) => {
  switch (action.type) {
    case GET_MEDICAL_BILLS_REQUEST:
      return { ...state, isLoading: true };
    case GET_MEDICAL_BILLS_SUCCESS:
      return { ...state, medicalBills: action.payload, isLoading: false };
    case GET_MEDICAL_BILLS_FAILURE:
      return { ...state, error: action.payload, isLoading: false };
    case CREATE_MEDICAL_BILL_REQUEST:
      return { ...state, isLoading: true };
    case CREATE_MEDICAL_BILL_SUCCESS:
      return { ...state, medicalBills: [...state.medicalBills, action.payload], isLoading: false };
    case CREATE_MEDICAL_BILL_FAILURE:
      return { ...state, error: action.payload, isLoading: false };
    case UPDATE_MEDICAL_BILL_REQUEST:
      return { ...state, isLoading: true };
    case UPDATE_MEDICAL_BILL_SUCCESS:
      return {
        ...state,
        medicalBills: state.medicalBills.map((medicalBill) =>
          medicalBill.id === action.payload.id ? action.payload : medicalBill
        ),
        isLoading: false,
      };
    case UPDATE_MEDICAL_BILL_FAILURE:
      return { ...state, error: action.payload, isLoading: false };
    case DELETE_MEDICAL_BILL_REQUEST:
      return { ...state, isLoading: true };
    case DELETE_MEDICAL_BILL_SUCCESS:
      return {
        ...state,
        medicalBills: state.medicalBills.filter((medicalBill) => medicalBill.id !== action.payload),
        isLoading: false,
      };
    case DELETE_MEDICAL_BILL_FAILURE:
      return { ...state, error: action.payload, isLoading: false };
    default:
      return state;
  }
};

export default combineReducers({ medicalBills: medicalBillsReducer
