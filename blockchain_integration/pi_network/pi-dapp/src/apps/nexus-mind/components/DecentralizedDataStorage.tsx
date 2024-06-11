import React, { useState, useEffect } from 'react';
import { IPFS } from 'ipfs-http-client';
import { DecentralizedDataStorageAPI } from '../api/decentralized-data-storage';

interface DecentralizedDataStorageProps {
  user: any;
}

const DecentralizedDataStorage: React.FC<DecentralizedDataStorageProps> = ({ user }) => {
  const [storedData, setStoredData] = useState({});

  useEffect(() => {
    const ipfs = new IPFS();
    const decentralizedDataStorageAPI = new DecentralizedDataStorageAPI();

    ipfs.add(user.data).then((hash) => {
      decentralizedDataStorageAPI.storeData(user.id, hash).then((storedData) => {
        setStoredData(storedData);
      });
    });
  }, [user]);

  return (
    <div>
      <h2>Decentralized Data Storage</h2>
      <p>Stored Data: {JSON.stringify(storedData)}</p>
    </div>
  );
};

export default DecentralizedDataStorage;
