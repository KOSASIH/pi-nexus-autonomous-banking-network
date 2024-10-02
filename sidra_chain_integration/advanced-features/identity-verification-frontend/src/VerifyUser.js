// src/VerifyUser.js
import React, { useState } from 'react';
import axios from 'axios';

const VerifyUser = () => {
  const [userAddress, setUserAddress] = useState('');
  const [userData, setUserData] = useState({});
  const [verified, setVerified] = useState(false);

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setUserData((prevUserData) => ({ ...prevUserData, [name]: value }));
  };

  const handleVerify = async () => {
    try {
      const response = await axios.post('http://localhost:3001/api/verify', {
        userAddress,
        userData,
      });
      setVerified(response.data.verified);
    } catch (error) {
      console.error(`Error verifying user: ${error}`);
    }
  };

  return (
    <div>
      <h1>Identity Verification</h1>
      <form>
        <label>
          User Address:
          <input
            type="text"
            value={userAddress}
            onChange={(event) => setUserAddress(event.target.value)}
          />
        </label>
        <br />
        <label>
          First Name:
          <input
            type="text"
            name="firstName"
            value={userData.firstName}
            onChange={handleInputChange}
          />
        </label>
        <br />
        <label>
          Last Name:
          <input
            type="text"
            name="lastName"
            value={userData.lastName}
            onChange={handleInputChange}
          />
        </label>
        <br />
        <label>
          Date of Birth:
          <input
            type="date"
            name="dateOfBirth"
            value={userData.dateOfBirth}
            onChange={handleInputChange}
          />
        </label>
        <br />
        <button onClick={handleVerify}>Verify</button>
      </form>
      {verified ? <p>User is verified!</p> : <p>User is not verified.</p>}
    </div>
  );
};

export default VerifyUser;
