import React, { useState, useEffect } from 'react';
import { uPort } from 'uport-credentials';
import { Ethereum } from 'ethereumjs-tx';

interface DecentralizedIdentityProps {
  user: any;
}

const DecentralizedIdentity: React.FC<DecentralizedIdentityProps> = ({ user }) => {
  const [did, setDid] = useState('');
  const [credentials, setCredentials] = useState({});

  useEffect(() => {
    const uport = new uPort();
    const ethereum = new Ethereum();

    uport.createIdentity(user.did).then((identity) => {
      setDid(identity.did);
    });

    uport.getCredentials(did).then((credentials) => {
      setCredentials(credentials);
    });

    ethereum.getTransactionCount(did).then((count) => {
      console.log(`Transaction count: ${count}`);
    });
  }, [user]);

  return (
    <div>
      <h2>Decentralized Identity</h2>
      <p>DID: {did}</p>
      <p>Credentials: {JSON.stringify(credentials)}</p>
    </div>
  );
};

export default DecentralizedIdentity;
