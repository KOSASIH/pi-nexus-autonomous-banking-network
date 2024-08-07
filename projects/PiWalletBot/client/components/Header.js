import React from 'react';
import styled from 'styled-components';
import { useAuth } from '../context/auth';

const HeaderContainer = styled.header`
  background-color: #333;
  color: #fff;
  padding: 20px;
  text-align: center;
`;

const Header = () => {
  const auth = useAuth();

  return (
    <HeaderContainer>
      <h1>Pi Wallet Bot</h1>
      {auth.isLoggedIn ? (
        <p>Welcome, {auth.user.username}!</p>
      ) : (
        <p>You are not logged in.</p>
      )}
    </HeaderContainer>
  );
};

export default Header;
