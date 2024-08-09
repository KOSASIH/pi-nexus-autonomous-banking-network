import React, { useState, useEffect, useContext } from 'react';
import { BrowserRouter, Route, Switch, Redirect } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { AuthContext } from '../contexts/AuthContext';
import { theme } from '../styles/theme';
import { GlobalStyle } from '../styles/global';
import Login from '../components/Login';
import Dashboard from '../components/Dashboard';
import NotFound from '../components/NotFound';

const App: React.FC = () => {
  const { accessToken, refreshToken, logout } = useContext(AuthContext);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    if (accessToken && refreshToken) {
      setIsAuthenticated(true);
    } else {
      setIsAuthenticated(false);
    }
  }, [accessToken, refreshToken]);

  const handleLogout = () => {
    logout();
    setIsAuthenticated(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <ToastContainer />
      <BrowserRouter>
        <Switch>
          <Route path="/login" exact>
            {isAuthenticated ? <Redirect to="/dashboard" /> : <Login />}
          </Route>
          <Route path="/dashboard" exact>
            {isAuthenticated ? <Dashboard /> : <Redirect to="/login" />}
          </Route>
          <Route path="*">
            <NotFound />
          </Route>
        </Switch>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
