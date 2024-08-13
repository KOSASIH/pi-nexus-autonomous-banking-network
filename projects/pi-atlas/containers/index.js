import React from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { AppContextProvider } from '../context/AppContext';
import { ClusterContainer } from './ClusterContainer';
import { EdgeContainer } from './EdgeContainer';
import { NodeContainer } from './NodeContainer';
import { Dashboard } from './Dashboard';
import { Navigation } from './Navigation';

const Containers = () => {
  return (
    <AppContextProvider>
      <BrowserRouter>
        <Navigation />
        <Switch>
          <Route path="/" exact component={Dashboard} />
          <Route path="/clusters/:clusterId" component={ClusterContainer} />
          <Route path="/edges/:edgeId" component={EdgeContainer} />
          <Route path="/nodes/:nodeId" component={NodeContainer} />
        </Switch>
      </BrowserRouter>
    </AppContextProvider>
  );
};

export default Containers;
