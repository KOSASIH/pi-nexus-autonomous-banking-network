import React, { useState, useEffect } from 'react';
import { uPort } from 'uport-credentials';
import { DecentralizedIdentityManagementAPI } from '../api/decentralized-identity-management';

interface DecentralizedIdentityManagementProps {
  user: any;
}

const DecentralizedIdentityManagement: React.FC<DecentralizedIdentityManagementProps> = ({ user }) => {
  const [identity, setIdentity] = useState({});

  useEffect(() => {
    const uport = new uPort();
    const decentralizedIdentityManagementAPI = new DecentralizedIdentityManagementAPI();

    uport.createIdentity(user.did).then((identity) => {
      setIdentity(identity);
    });

    decentralizedIdentityManagementAPI.getIdentity(user.id).then((identity) => {
      setIdentity(identity);
    });
  }, [user]);

  return (
    <div>
      <h2>Decentralized Identity Management</h2>
      <p>Identity: {JSON.stringify(identity)}</p>
    </div>
  );
};

export default DecentralizedIdentityManagement;
