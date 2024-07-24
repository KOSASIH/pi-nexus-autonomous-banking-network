import React from 'react';
import { Provider } from 'react-redux';
import { createStore, combineReducers } from 'redux';
import voteReducer from '../reducers/vote.reducer';
import VoteComponent from '../components/VoteComponent';

const rootReducer = combineReducers({ vote: voteReducer });
const store = createStore(rootReducer);

const AppContainer = () => {
  return (
    <Provider store={store}>
      <VoteComponent />
    </Provider>
  );
};

export default AppContainer;
