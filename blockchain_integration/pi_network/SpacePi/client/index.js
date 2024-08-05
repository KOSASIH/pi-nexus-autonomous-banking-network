import React from 'eact';
import ReactDOM from 'eact-dom';
import App from './containers/App';
import store from './store';

ReactDOM.render(
  <React.StrictMode>
    <App store={store} />
  </React.StrictMode>,
  document.getElementById('root')
);
