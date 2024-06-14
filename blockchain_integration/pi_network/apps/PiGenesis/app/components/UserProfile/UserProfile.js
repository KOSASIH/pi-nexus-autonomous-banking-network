import React, { useState, useEffect } from 'eact';
import axios from 'axios';

const UserProfile = () => {
  const [user, setUser] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('/api/user/profile')
     .then(response => {
        setUser(response.data);
        setLoading(false);
      })
     .catch(error => {
        console.error(error);
      });
  }, []);

  const handleUpdateProfile = (updates) => {
    axios.patch('/api/user/profile', updates)
     .then(response => {
        setUser(response.data);
      })
     .catch(error => {
        console.error(error);
      });
  };

  return (
    <div>
      {loading? (
        <p>Loading...</p>
      ) : (
        <div>
          <h2>{user.name}</h2>
          <p>Email: {user.email}</p>
          <p>Phone: {user.phone}</p>
          <button onClick={() => handleUpdateProfile({ name: 'New Name' })}>
            Update Profile
          </button>
        </div>
      )}
    </div>
  );
};

export default UserProfile;
