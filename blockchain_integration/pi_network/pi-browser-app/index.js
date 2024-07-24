import React from 'react';
import ReactDOM from 'react-dom';
import App from './containers/App';
import PiBrowserBlockchainExplorer from './components/PiBrowserBlockchainExplorer';

ReactDOM.render(
  <React.StrictMode>
    <App>
      <PiBrowserBlockchainExplorer />
    </App>
  </React.StrictMode>,
  document.getElementById('root')
);
