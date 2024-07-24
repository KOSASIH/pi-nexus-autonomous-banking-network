import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { AppContainer } from 'react-hot-loader';

import App from './App';
import reducers from './reducers';
import sagas from './sagas';
import { configureStore } from './store';

const store = configureStore();

ReactDOM.render(
  <AppContainer>
    <Provider store={store}>
      <BrowserRouter>
        <Switch>
          <Route path="/" component={App} />
        </Switch>
      </BrowserRouter>
    </Provider>
  </AppContainer>,
  document.getElementById('root')
);
