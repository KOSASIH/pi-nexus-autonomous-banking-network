import React, { useState, useEffect } from 'react';
import { Ethereum } from 'ethereumjs-tx';
import { IdentityVerificationAPI } from '../api/identity-verification';

interface BlockchainBasedIdentityVerificationProps {
  user: any;
}

const BlockchainBasedIdentityVerification: React.FC<BlockchainBasedIdentityVerificationProps> = ({ user }) => {
  const [identityVerified, setIdentityVerified] = useState(false);

  useEffect(() => {
    const ethereum = new Ethereum();
    const identityVerificationAPI = new IdentityVerificationAPI();

    ethereum.getTransactionCount(user.id).then((count) => {
      if (count > 0) {
        setIdentityVerified(true);
      }
    });

    identityVerificationAPI.verifyIdentity(user.id).then((verified) => {
      setIdentityVerified(verified);
    });
  }, [user]);

  return (
    <div>
      <h2>Blockchain-based Identity Verification</h2>
      <p>Identity Verified: {identityVerified ? 'Yes' : 'No'}</p>
    </div>
  );
};

export default BlockchainBasedIdentityVerification;
