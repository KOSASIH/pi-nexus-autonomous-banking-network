import { combineReducers } from 'redux';
import { createReducer } from '@reduxjs/toolkit';
import {
  NODES_FETCH_REQUEST,
  NODES_FETCH_SUCCESS,
  NODES_FETCH_FAILURE,
  NODE_SELECT,
  NODE_DESELECT,
  NODE_INCENTIVIZATION_UPDATE,
  NODE_REPUTATION_UPDATE,
} from '../actions/nodeActions';

const initialState = {
  nodes: [],
  selectedNodes: [],
  incentivizationData: {},
  reputationData: {},
  isLoading: false,
  error: null,
};

const nodesReducer = createReducer(initialState, {
  [NODES_FETCH_REQUEST]: (state) => {
    state.isLoading = true;
  },
  [NODES_FETCH_SUCCESS]: (state, action) => {
    state.isLoading = false;
    state.nodes = action.payload.nodes;
    state.incentivizationData = action.payload.incentivization;
    state.reputationData = action.payload.reputation;
  },
  [NODES_FETCH_FAILURE]: (state, action) => {
    state.isLoading = false;
    state.error = action.payload.error;
  },
  [NODE_SELECT]: (state, action) => {
    state.selectedNodes = [...state.selectedNodes, action.payload.nodeId];
  },
  [NODE_DESELECT]: (state, action) => {
    state.selectedNodes = state.selectedNodes.filter((id) => id !== action.payload.nodeId);
  },
  [NODE_INCENTIVIZATION_UPDATE]: (state, action) => {
    state.incentivizationData = action.payload.incentivizationData;
  },
  [NODE_REPUTATION_UPDATE]: (state, action) => {
    state.reputationData = action.payload.reputationData;
  },
});

export default combineReducers({
  nodes: nodesReducer,
});
