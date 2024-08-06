import { combineReducers } from 'redux';
import { createReducer } from '@reduxjs/toolkit';
import { GET_MEDICAL_BILLS, GET_MEDICAL_BILL, CREATE_MEDICAL_BILL, UPDATE_MEDICAL_BILL, DELETE_MEDICAL_BILL } from '../medical-billing-actions';

const initialState = {
  medicalBills: [],
  medicalBill: null,
  loading: false,
  error: null,
};

const medicalBillsReducer = createReducer(initialState, {
  [GET_MEDICAL_BILLS.pending]: (state) => {
    state.loading = true;
  },
  [GET_MEDICAL_BILLS.fulfilled]: (state, action) => {
    state.medicalBills = action.payload;
    state.loading = false;
  },
  [GET_MEDICAL_BILLS.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
  [GET_MEDICAL_BILL.pending]: (state) => {
    state.loading = true;
  },
  [GET_MEDICAL_BILL.fulfilled]: (state, action) => {
    state.medicalBill = action.payload;
    state.loading = false;
  },
  [GET_MEDICAL_BILL.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
  [CREATE_MEDICAL_BILL.pending]: (state) => {
    state.loading = true;
  },
  [CREATE_MEDICAL_BILL.fulfilled]: (state, action) => {
    state.medicalBills = [...state.medicalBills, action.payload];
    state.loading = false;
  },
  [CREATE_MEDICAL_BILL.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
  [UPDATE_MEDICAL_BILL.pending]: (state) => {
    state.loading = true;
  },
    [UPDATE_MEDICAL_BILL.fulfilled]: (state, action) => {
    const index = state.medicalBills.findIndex((bill) => bill.id === action.payload.id);
    state.medicalBills[index] = action.payload;
    state.loading = false;
  },
  [UPDATE_MEDICAL_BILL.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
  [DELETE_MEDICAL_BILL.pending]: (state) => {
    state.loading = true;
  },
  [DELETE_MEDICAL_BILL.fulfilled]: (state, action) => {
    state.medicalBills = state.medicalBills.filter((bill) => bill.id !== action.payload.id);
    state.loading = false;
  },
  [DELETE_MEDICAL_BILL.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
});

export default combineReducers({
  medicalBills: medicalBillsReducer,
});
   
