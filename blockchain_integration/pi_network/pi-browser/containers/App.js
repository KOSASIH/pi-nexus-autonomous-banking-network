// containers/App.js
import React from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import Sidebar from '../components/Sidebar';
import Browser from './Browser';
import Wallet from './Wallet';

const App = () => {
  return (
    <BrowserRouter>
      <div className="app">
        <Header />
        <Sidebar />
        <div className="content">
          <Switch>
            <Route path="/" exact component={Browser} />
            <Route path="/wallet" component={Wallet} />
          </Switch>
        </div>
        <Footer />
      </div>
    </BrowserRouter>
  );
};

export default App;
