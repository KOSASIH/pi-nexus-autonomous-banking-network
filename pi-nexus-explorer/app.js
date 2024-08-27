import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { Provider } from 'react-redux';
import store from './store';
import BlockContainer from './containers/BlockContainer';
import TransactionContainer from './containers/TransactionContainer';
import ContractContainer from './containers/ContractContainer';
import './styles/index.css';

const App = () => {
  return (
    <Provider store={store}>
      <BrowserRouter>
        <Switch>
          <Route path="/" exact component={BlockContainer} />
          <Route path="/transactions" component={TransactionContainer} />
          <Route path="/contracts" component={ContractContainer} />
        </Switch>
      </BrowserRouter>
    </Provider>
  );
};

ReactDOM.render(<App />, document.getElementById('root'));
