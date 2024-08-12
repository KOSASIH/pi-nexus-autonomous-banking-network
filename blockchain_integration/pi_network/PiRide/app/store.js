import { createStore, combineReducers, applyMiddleware } from 'redux';
import { composeWithDevTools } from 'redux-devtools-extension';
import thunk from 'redux-thunk';
import { rideReducer } from './reducers/rideReducer';
import { userReducer } from './reducers/userReducer';
import { notificationReducer } from './reducers/notificationReducer';

const rootReducer = combineReducers({
  ride: rideReducer,
  user: userReducer,
  notification: notificationReducer,
});

const middleware = [thunk];

const store = createStore(
  rootReducer,
  composeWithDevTools(applyMiddleware(...middleware))
);

export default store;
