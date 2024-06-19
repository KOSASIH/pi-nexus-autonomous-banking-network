import React, { useState, useEffect } from 'react';
import Web3 from 'web3';
import { useContractLoader, useContractReader } from 'eth-hooks';

const Shipment = ({ shipmentId, provider, blockNumber }) => {
  const [status, setStatus] = useState('');
  const contract = useContractLoader('MyContract', provider);

  useEffect(() => {
    async function fetchData() {
      const shipment = await contract.methods.shipments(shipmentId).call();
      setStatus(shipment.status);
    }
    fetchData();
  }, [shipmentId, blockNumber]);

  return (
    <div>
      <h2>Shipment: {shipmentId}</h2>
      <p>Status: {status}</p>
    </div>
  );
};

export default Shipment;
