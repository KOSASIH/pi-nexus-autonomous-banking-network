import React from 'react';
import styled from 'styled-components';

const SidebarContainer = styled.div`
  background-color: #333;
  color: #fff;
  padding: 20px;
  width: 200px;
  height: 100vh;
  position: fixed;
  top: 0;
  left: 0;
`;

const Sidebar = () => {
  return (
    <SidebarContainer>
      <h2>Menu</h2>
      <ul>
        <li><a href="#">Dashboard</a></li>
        <li><a href="#">Transactions</a></li>
        <li><a href="#">Settings</a></li>
      </ul>
    </SidebarContainer>
  );
};

export default Sidebar;
