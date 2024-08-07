import React from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import InsuranceForm from '../components/InsuranceForm';
import PolicyList from '../components/PolicyList';

const AppContainer = () => {
  return (
    <BrowserRouter>
      <Switch>
        <Route path="/" exact component={InsuranceForm} />
        <Route path="/policies" component={PolicyList} />
      </Switch>
    </BrowserRouter>
  );
};

export default AppContainer;
