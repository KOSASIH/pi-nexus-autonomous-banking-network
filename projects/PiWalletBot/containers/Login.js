import React, { useState } from 'react';
import { useAuth } from '../context/auth';
import { useApi } from '../context/api';
import LoginForm from '../components/LoginForm';

const Login = () => {
  const auth = useAuth();
  const api = useApi();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);

  const handleLogin = async (event) => {
    event.preventDefault();
    try {
      const response = await api.login(username, password);
      auth.login(response.data.user);
    } catch (error) {
      setError(error.message);
    }
  };

  return (
    <div className="login-container">
      <h2>Login</h2>
      <LoginForm
        username={username}
        password={password}
        error={error}
        onUsernameChange={(event) => setUsername(event.target.value)}
        onPasswordChange={(event) => setPassword(event.target.value)}
        onLogin={handleLogin}
      />
    </div>
  );
};

export default Login;
