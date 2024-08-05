import React from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import Header from '../components/Header';
import Markets from '../containers/Markets';
import Trade from '../containers/Trade';

const App = () => {
  return (
    <BrowserRouter>
      <Header />
      <Switch>
        <Route path="/" exact component={Markets} />
        <Route path="/trade" component={Trade} />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
