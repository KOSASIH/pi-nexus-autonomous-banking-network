import React from 'eact';
import ReactDOM from 'eact-dom';
import { Provider } from 'eact-redux';
import { createStore, combineReducers } from 'edux';
import { BrowserRouter, Route, Switch } from 'eact-router-dom';
import LogisticsContainer from './containers/LogisticsContainer';
import ShipmentContainer from './containers/ShipmentContainer';
import logisticsReducer from './reducers/logisticsReducer';
import shipmentReducer from './reducers/shipmentReducer';

const rootReducer = combineReducers({
  logistics: logisticsReducer,
  shipments: shipmentReducer,
});

const store = createStore(rootReducer);

ReactDOM.render(
  <Provider store={store}>
    <BrowserRouter>
      <Switch>
        <Route path="/" exact component={LogisticsContainer} />
        <Route path="/shipments/:id" component={ShipmentContainer} />
      </Switch>
    </BrowserRouter>
  </Provider>,
  document.getElementById('root')
);
