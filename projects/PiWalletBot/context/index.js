import React from 'react';
import ReactDOM from 'react-dom';
import { AuthProvider } from './context/auth';
import { WalletProvider } from './context/wallet';
import App from './App';

ReactDOM.render(
  <React.StrictMode>
    <AuthProvider>
      <WalletProvider>
        <App />
      </WalletProvider>
    </AuthProvider>
  </React.StrictMode>,
  document.getElementById('root')
);
