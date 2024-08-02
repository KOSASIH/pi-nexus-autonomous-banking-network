// index.js
import React from 'eact';
import ReactDOM from 'eact-dom';
import { Web3Provider } from './contexts/Web3Context';
import { PaymentGatewayProvider } from './contexts/PaymentGatewayContext';
import App from './App';
import './styles/globals.css';

ReactDOM.render(
  <React.StrictMode>
    <Web3Provider>
      <PaymentGatewayProvider>
        <App />
      </PaymentGatewayProvider>
    </Web3Provider>
  </React.StrictMode>,
  document.getElementById('root')
);
