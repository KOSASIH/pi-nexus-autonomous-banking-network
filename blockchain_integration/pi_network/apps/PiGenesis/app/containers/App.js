import React, { useState, useEffect } from 'eact';
import { BrowserRouter, Route, Switch } from 'eact-router-dom';
import Header from '../components/Header';
import Dashboard from './Dashboard';
import Portfolio from './Portfolio';
import Settings from './Settings';

const App = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    // Initialize user data from local storage or API
  }, []);

  return (
    <BrowserRouter>
      <Header />
      <Switch>
        <Route path="/" exact component={Dashboard} />
        <Route path="/portfolio" component={Portfolio} />
        <Route path="/settings" component={Settings} />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
