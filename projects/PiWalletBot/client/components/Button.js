import React from 'react';
import styled from 'styled-components';

const ButtonContainer = styled.button`
  background-color: #4CAF50;
  color: #fff;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;

  &:hover {
    background-color: #3e8e41;
  }
`;

const Button = ({ children, onClick, disabled }) => {
  return (
    <ButtonContainer onClick={onClick} disabled={disabled}>
      {children}
    </ButtonContainer>
  );
};

export default Button;
