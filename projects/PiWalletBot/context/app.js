import React from 'react';
import { useState, useEffect } from 'react';
import { AuthContext } from './context/auth';
import { WalletContext } from './context/wallet';
import Login from './containers/Login';
import Register from './containers/Register';
import Dashboard from './containers/Dashboard';

const App = () => {
  const { user, token, login, register, logout } = React.useContext(AuthContext);
  const { balance, transactions, getBalance, getTransactions } = React.useContext(WalletContext);
  const [showLogin, setShowLogin] = useState(true);

  useEffect(() => {
    if (token) {
      setShowLogin(false);
    }
  }, [token]);

  return (
    <div className="app">
      {showLogin ? (
        <Login login={login} />
      ) : (
        <div>
          <Register register={register} />
          <button onClick={logout}>Logout</button>
        </div>
      )}
      {user && (
        <Dashboard
          user={user}
          balance={balance}
          transactions={transactions}
          getBalance={getBalance}
          getTransactions={getTransactions}
        />
      )}
    </div>
  );
};

export default App;
