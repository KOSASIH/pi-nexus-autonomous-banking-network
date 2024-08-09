import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import axios from 'axios';
import { useHistory } from 'react-router-dom';

interface LoginForm {
  email: string;
  password: string;
}

const schema = yup.object().shape({
  email: yup.string().email('Invalid email').required('Email is required'),
  password: yup.string().min(8, 'Password must be at least 8 characters').required('Password is required'),
});

const Login: React.FC = () => {
  const { register, handleSubmit, errors } = useForm<LoginForm>({
    resolver: yupResolver(schema),
  });
  const [loading, setLoading] = useState(false);
  const history = useHistory();

  const onSubmit = async (data: LoginForm) => {
    try {
      setLoading(true);
      const response = await axios.post('/api/auth/login', data);
      const { accessToken, refreshToken } = response.data;
      localStorage.setItem('accessToken', accessToken);
      localStorage.setItem('refreshToken', refreshToken);
      history.push('/dashboard');
    } catch (error) {
      console.error(error);
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Login</h1>
      <form onSubmit={handleSubmit(onSubmit)}>
        <label>Email:</label>
        <input type="email" {...register('email')} />
        {errors.email && <div>{errors.email.message}</div>}
        <br />
        <label>Password:</label>
        <input type="password" {...register('password')} />
        {errors.password && <div>{errors.password.message}</div>}
        <br />
        <button type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Login'}
        </button>
      </form>
    </div>
  );
};

export default Login;
