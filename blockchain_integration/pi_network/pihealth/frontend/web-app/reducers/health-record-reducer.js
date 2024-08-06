import { combineReducers } from 'redux';
import { createReducer } from '@reduxjs/toolkit';
import { GET_HEALTH_RECORDS, GET_HEALTH_RECORD, CREATE_HEALTH_RECORD, UPDATE_HEALTH_RECORD, DELETE_HEALTH_RECORD } from '../health-record-actions';

const initialState = {
  healthRecords: [],
  healthRecord: null,
  loading: false,
  error: null,
};

const healthRecordsReducer = createReducer(initialState, {
  [GET_HEALTH_RECORDS.pending]: (state) => {
    state.loading = true;
  },
  [GET_HEALTH_RECORDS.fulfilled]: (state, action) => {
    state.healthRecords = action.payload;
    state.loading = false;
  },
  [GET_HEALTH_RECORDS.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
  [GET_HEALTH_RECORD.pending]: (state) => {
    state.loading = true;
  },
  [GET_HEALTH_RECORD.fulfilled]: (state, action) => {
    state.healthRecord = action.payload;
    state.loading = false;
  },
  [GET_HEALTH_RECORD.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
  [CREATE_HEALTH_RECORD.pending]: (state) => {
    state.loading = true;
  },
  [CREATE_HEALTH_RECORD.fulfilled]: (state, action) => {
    state.healthRecords = [...state.healthRecords, action.payload];
    state.loading = false;
  },
  [CREATE_HEALTH_RECORD.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
  [UPDATE_HEALTH_RECORD.pending]: (state) => {
    state.loading = true;
  },
  [UPDATE_HEALTH_RECORD.fulfilled]: (state, action) => {
    const index = state.healthRecords.findIndex((record) => record.id === action.payload.id);
    state.healthRecords[index] = action.payload;
    state.loading = false;
  },
  [UPDATE_HEALTH_RECORD.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
  [DELETE_HEALTH_RECORD.pending]: (state) => {
    state.loading = true;
  },
  [DELETE_HEALTH_RECORD.fulfilled]: (state, action) => {
    state.healthRecords = state.healthRecords.filter((record) => record.id !== action.payload.id);
    state.loading = false;
  },
  [DELETE_HEALTH_RECORD.rejected]: (state, action) => {
    state.error = action.error;
    state.loading = false;
  },
});

export default combineReducers({
  healthRecords: healthRecordsReducer,
});
