import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './components/Auth/Login';
import Register from './components/Auth/Register';
import PasswordReset from './components/Auth/PasswordReset';
import TransactionList from './components/Transactions/TransactionList';

const App = () => {
    return (
        <Router>
            <Navbar />
            <Switch>
                <Route path="/login" component={Login} />
                <Route path="/register" component={Register} />
                <Route path="/password-reset" component={PasswordReset} />
                <Route path="/transactions" component={TransactionList} />
                <Route path="/" exact>
                    <h1>Welcome to Pi Nexus Autonomous Banking Network</h1>
                </Route>
            </Switch>
        </Router>
    );
};

export default App;
