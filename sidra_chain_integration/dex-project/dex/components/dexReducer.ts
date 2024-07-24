import { createReducer } from '@reduxjs/toolkit';
import { fetchDexData, trade } from './dexActions';

const initialState = {
  data: [],
  loading: false,
};

const dexReducer = createReducer(initialState, {
  [fetchDexData.pending]: (state) => {
    state.loading = true;
  },
  [fetchDexData.fulfilled]: (state, action) => {
    state.data = action.payload;
    state.loading = false;
  },
  [fetchDexData.rejected]: (state) => {
    state.loading = false;
  },
  [trade.pending]: (state) => {
    state.loading = true;
  },
  [trade.fulfilled]: (state, action) => {
    state.data = action.payload;
    state.loading = false;
  },
  [trade.rejected]: (state) => {
    state.loading = false;
  },
});

export default dexReducer;
