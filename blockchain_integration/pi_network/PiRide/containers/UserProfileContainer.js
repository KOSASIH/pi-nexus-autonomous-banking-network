import React, { useState, useEffect, useContext } from 'react';
import { useWeb3 } from '@pi-network/web3-react';
import { RideContract } from '../blockchain/smartContracts/RideContract';
import { UserProfileForm } from '../components/UserProfileForm';
import { UserProfileCard } from '../components/UserProfileCard';
import { NotificationContext } from '../contexts/NotificationContext';

const UserProfileContainer = () => {
  const { account, library } = useWeb3();
  const [user, setUser] = useState({});
  const [loading, setLoading] = useState(false);
  const { notify } = useContext(NotificationContext);

  useEffect(() => {
    const rideContract = new RideContract(library);
    rideContract.getUserData(account).then((data) => setUser(data));
  }, [account, library]);

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const rideContract = new RideContract(library);
      const txHash = await rideContract.updateUserProfile(
        account,
        user.name,
        user.email,
        user.phone,
        user.address
      );
      notify(`User profile updated successfully! Tx Hash: ${txHash}`);
      setLoading(false);
    } catch (error) {
      notify(`Error updating user profile: ${error.message}`);
      setLoading(false);
    }
  };

  const handleNameChange = (name) => {
    setUser({ ...user, name });
  };

  const handleEmailChange = (email) => {
    setUser({ ...user, email });
  };

  const handlePhoneChange = (phone) => {
    setUser({ ...user, phone });
  };

  const handleAddressChange = (address) => {
    setUser({ ...user, address });
  };

  return (
    <div>
      <h2>User Profile Container</h2>
      <UserProfileForm
        onSubmit={handleFormSubmit}
        name={user.name}
        email={user.email}
        phone={user.phone}
        address={user.address}
        onChangeName={handleNameChange}
        onChangeEmail={handleEmailChange}
        onChangePhone={handlePhoneChange}
        onChangeAddress={handleAddressChange}
      />
      <UserProfileCard user={user} />
      {loading && <p>Loading...</p>}
    </div>
  );
};

export default UserProfileContainer;
