import React from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import Launches from './Launches';
import Merchandise from './Merchandise';
import Users from './Users';

const App = () => {
  return (
    <BrowserRouter>
      <Header />
      <Switch>
        <Route path="/" exact component={Launches} />
        <Route path="/launches" component={Launches} />
        <Route path="/merchandise" component={Merchandise} />
        <Route path="/users" component={Users} />
      </Switch>
      <Footer />
    </BrowserRouter>
  );
};

export default App;
