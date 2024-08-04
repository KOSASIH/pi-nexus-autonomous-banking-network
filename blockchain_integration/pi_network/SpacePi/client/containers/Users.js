import React, { useState, useEffect } from 'eact';
import axios from 'axios';
import UserCard from '../components/UserCard';

const Users = () => {
  const [users, setUsers] = useState([]);

  useEffect(() => {
    axios.get('/api/users')
     .then(response => {
        setUsers(response.data);
      })
     .catch(error => {
        console.error(error);
      });
  }, []);

  return (
    <div className="users">
      <h1>Users</h1>
      <ul>
        {users.map(user => (
          <li key={user.id}>
            <UserCard user={user} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Users;
