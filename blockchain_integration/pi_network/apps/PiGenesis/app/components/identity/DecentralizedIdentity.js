import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';
import { useBiometricAuth } from '@pi-nexus/biometric-auth-react';

const DecentralizedIdentity = () => {
  const [identityData, setIdentityData] = useState(null);
  const { blockchain } = useBlockchain();
  const { biometricAuth } = useBiometricAuth();

  useEffect(() => {
    const fetchIdentityData = async () => {
      const data = await blockchain.getIdentityData();
      setIdentityData(data);
    };

    fetchIdentityData();
  }, [blockchain]);

  const handleUpdateIdentity = async (newData) => {
    const updatedData = await blockchain.updateIdentityData(newData);
    setIdentityData(updatedData);
  };

  const handleBiometricAuth = async () => {
    const authResult = await biometricAuth.authenticate();
    if (authResult) {
      // Authenticate and authorize access to identity data
      const identityData = await blockchain.getIdentityData();
      setIdentityData(identityData);
    }
  };

  return (
    <div>
      <h1>Decentralized Identity Management</h1>
      {identityData && (
        <IdentityViewer data={identityData} />
      )}
      <button onClick={() => handleUpdateIdentity({ name: 'John Doe', age: 30 })}>Update Identity</button>
      <button onClick={handleBiometricAuth}>Authenticate with Biometrics</button>
    </div>
  );
};

export default DecentralizedIdentity;
