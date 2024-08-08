import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as formReducer } from 'redux-form';
import { apiReducer } from './api/reducer';
import { authReducer } from './auth/reducer';
import { walletReducer } from './wallet/reducer';
import App from './App';
import './styles/globals.css';
import './styles/index.css';

const rootReducer = combineReducers({
  form: formReducer,
  api: apiReducer,
  auth: authReducer,
  wallet: walletReducer,
});

const store = createStore(rootReducer);

ReactDOM.render(
  <Provider store={store}>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </Provider>,
  document.getElementById('root')
);
