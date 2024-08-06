import React from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import Nav from './components/Nav';
import Login from './components/Login';
import Dashboard from './components/Dashboard';

const App = () => {
  return (
    <BrowserRouter>
      <Nav />
      <Switch>
        <Route path="/" exact component={Dashboard} />
        <Route path="/login" component={Login} />
        <Route path="/medical-billings" component={() => <div>Medical Billings page</div>} />
        <Route path="/patients" component={() => <div>Patients page</div>} />
        <Route path="/healthcare-providers" component={() => <div>Healthcare Providers page</div>} />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
