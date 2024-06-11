import React, { useState, useEffect } from 'react';
import { Ethereum } from 'ethereumjs-tx';
import { KYCVerificationAPI } from '../api/kyc-verification';

interface BlockchainBasedKYCVerificationProps {
  user: any;
}

const BlockchainBasedKYCVerification: React.FC<BlockchainBasedKYCVerificationProps> = ({ user }) => {
  const [kycVerified, setKycVerified] = useState(false);

  useEffect(() => {
    const ethereum = new Ethereum();
    const kycVerificationAPI = new KYCVerificationAPI();

    ethereum.getTransactionCount(user.id).then((count) => {
      if (count > 0) {
        setKycVerified(true);
      }
    });

    kycVerificationAPI.verifyKYC(user.id).then((verified) => {
      setKycVerified(verified);
    });
  }, [user]);

  return (
    <div>
      <h2>Blockchain-based KYC Verification</h2>
      <p>KYC Verified: {kycVerified ? 'Yes' : 'No'}</p>
    </div>
  );
};

export default BlockchainBasedKYCVerification;
