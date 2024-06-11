import React, { useState, useEffect } from 'eact';
import { NTRU } from 'ntru-js';

interface QuantumResistantCryptographyProps {
  user: any;
}

const QuantumResistantCryptography: React.FC<QuantumResistantCryptographyProps> = ({ user }) => {
  const [encryptedData, setEncryptedData] = useState('');
  const [decryptedData, setDecryptedData] = useState('');

  useEffect(() => {
    const ntru = new NTRU();

    ntru.generateKeyPair().then((keyPair) => {
      const publicKey = keyPair.publicKey;
      const privateKey = keyPair.privateKey;

      ntru.encrypt(user.data, publicKey).then((encryptedData) => {
        setEncryptedData(encryptedData);
      });

      ntru.decrypt(encryptedData, privateKey).then((decryptedData) => {
        setDecryptedData(decryptedData);
      });
    });
  }, [user]);

  return (
    <div>
      <h2>Quantum Resistant Cryptography</h2>
      <p>Encrypted Data: {encryptedData}</p>
      <p>Decrypted Data: {decryptedData}</p>
    </div>
  );
};

export default QuantumResistantCryptography;
