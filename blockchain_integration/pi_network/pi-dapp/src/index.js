import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { Web3Provider } from '@ethersproject/providers';
import { Web3ReactProvider } from '@web3-react/core';
import { PiNetworkProvider } from './PiNetworkProvider';
import App from './App';
import './index.css';

const web3Provider = new Web3Provider(window.ethereum);
const piNetworkProvider = new PiNetworkProvider(web3Provider);

ReactDOM.render(
  <React.StrictMode>
    <Web3ReactProvider>
      <PiNetworkProvider>
        <BrowserRouter>
          <Switch>
            <Route path="/" component={App} exact />
          </Switch>
        </BrowserRouter>
      </PiNetworkProvider>
    </Web3ReactProvider>
  </React.StrictMode>,
  document.getElementById('root')
);
