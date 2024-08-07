import React, { useState } from 'react';
import { useAuth } from '../context/auth';
import { useApi } from '../context/api';
import RegisterForm from '../components/RegisterForm';

const Register = () => {
  const auth = useAuth();
  const api = useApi();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState(null);

  const handleRegister = async (event) => {
    event.preventDefault();
    try {
      const response = await api.register(username, password, email);
      auth.register(response.data.user);
    } catch (error) {
      setError(error.message);
    }
  };

  return (
    <div className="register-container">
      <h2>Register</h2>
      <RegisterForm
        username={username}
        password={password}
        email={email}
        error={error}
        onUsernameChange={(event) => setUsername(event.target.value)}
        onPasswordChange={(event) => setPassword(event.target.value)}
        onEmailChange={(event) => setEmail(event.target.value)}
        onRegister={handleRegister}
      />
    </div>
  );
};

export default Register;
