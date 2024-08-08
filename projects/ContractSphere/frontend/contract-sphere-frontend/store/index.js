import { createStore, combineReducers } from 'redux';
import contractReducer from '../reducers/contractReducer';

const rootReducer = combineReducers({
  contract: contractReducer
});

const store = createStore(rootReducer);

export default store;
