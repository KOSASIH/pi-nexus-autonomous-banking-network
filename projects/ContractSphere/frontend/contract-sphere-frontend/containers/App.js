import React from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import Header from '../components/Header';
import ContractList from '../components/ContractList';
import ContractDetails from '../components/ContractDetails';

const App = () => {
  return (
    <BrowserRouter>
      <Header />
      <Switch>
        <Route path="/" exact component={ContractList} />
        <Route path="/contracts/:id" component={ContractDetails} />
        <Route path="/create-contract" component={CreateContract} />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
