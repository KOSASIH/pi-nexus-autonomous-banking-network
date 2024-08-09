import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react';
import { Login } from './Login';

describe('Login component', () => {
  it('renders correctly', () => {
    const { getByText } = render(<Login />);
    expect(getByText('Login')).toBeInTheDocument();
  });

  it('calls login function on submit', () => {
    const loginFn = jest.fn();
    const { getByText } = render(<Login login={loginFn} />);
    const form = getByText('Login');
    fireEvent.submit(form);
    expect(loginFn).toHaveBeenCalledTimes(1);
  });

  it('displays error message on invalid credentials', async () => {
    const loginFn = jest.fn(() => Promise.reject(new Error('Invalid credentials')));
    const { getByText } = render(<Login login={loginFn} />);
    const form = getByText('Login');
    fireEvent.submit(form);
    await waitFor(() => getByText('Invalid credentials'));
    expect(getByText('Invalid credentials')).toBeInTheDocument();
  });
});
