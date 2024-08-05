import { createStore, combineReducers } from 'edux';
import reducers from './reducers';

const store = createStore(combineReducers(reducers));

export default store;
