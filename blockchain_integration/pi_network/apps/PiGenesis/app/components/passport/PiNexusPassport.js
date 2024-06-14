import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';
import { useBiometricAuth } from '@pi-nexus/biometric-auth-react';

const PiNexusPassport = () => {
  const [passportData, setPassportData] = useState(null);
  const { blockchain } = useBlockchain();
  const { biometricAuth } = useBiometricAuth();

  useEffect(() => {
    const fetchPassportData = async () => {
      const data = await blockchain.getPassportData();
      setPassportData(data);
    };

    fetchPassportData();
  }, [blockchain]);

  const handleBiometricAuth = async () => {
    const authResult = await biometricAuth.authenticate();
    if (authResult) {
      // Authenticate and authorize access to passport data
      const passportData = await blockchain.getPassportData();
      setPassportData(passportData);
    }
  };

  return (
    <div>
      <h1>Pi Nexus Passport</h1>
      {passportData && (
        <PassportDataViewer data={passportData} />
      )}
      <button onClick={handleBiometricAuth}>Authenticate with Biometrics</button>
    </div>
  );
};

export default PiNexusPassport;
