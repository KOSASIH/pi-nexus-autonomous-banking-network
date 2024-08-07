import React, { useState, useEffect } from 'react';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { useAuth } from '../context/auth';
import { useApi } from '../context/api';
import Header from '../components/Header';
import Sidebar from '../components/Sidebar';
import Dashboard from './Dashboard';
import Login from './Login';
import Register from './Register';
import Chatbot from './Chatbot';

const App = () => {
  const auth = useAuth();
  const api = useApi();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('token');
        if (token) {
          const response = await api.validateToken(token);
          if (response.data.isValid) {
            auth.login(response.data.user);
          }
        }
      } catch (error) {
        console.error(error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  if (loading) {
    return <Loading />;
  }

  return (
    <BrowserRouter>
      <Header />
      <Sidebar />
      <Switch>
        <Route path="/" exact component={Dashboard} />
        <Route path="/login" component={Login} />
        <Route path="/register" component={Register} />
        <Route path="/chatbot" component={Chatbot} />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
