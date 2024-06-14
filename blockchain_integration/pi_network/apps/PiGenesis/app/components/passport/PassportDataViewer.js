import { useState, useEffect } from 'eact';
import { useBlockchain } from '@pi-nexus/blockchain-react';

const PassportDataViewer = ({ data }) => {
  const [identity, setIdentity] = useState(null);
  const [credentials, setCredentials] = useState(null);
  const [assets, setAssets] = useState(null);
  const { blockchain } = useBlockchain();

  useEffect(() => {
    const fetchIdentity = async () => {
      const identityData = await blockchain.getIdentityData(data.identityId);
      setIdentity(identityData);
    };

    const fetchCredentials = async () => {
      const credentialsData = await blockchain.getCredentialsData(data.credentialsId);
      setCredentials(credentialsData);
    };

    const fetchAssets = async () => {
      const assetsData = await blockchain.getAssetsData(data.assetsId);
      setAssets(assetsData);
    };

    fetchIdentity();
    fetchCredentials();
    fetchAssets();
  }, [data, blockchain]);

  return (
    <div>
      <h2>Identity</h2>
      {identity && (
        <IdentityViewer data={identity} />
      )}
      <h2>Credentials</h2>
      {credentials && (
        <CredentialsViewer data={credentials} />
      )}
      <h2>Assets</h2>
      {assets && (
        <AssetsViewer data={assets} />
      )}
    </div>
  );
};

export default PassportDataViewer;
