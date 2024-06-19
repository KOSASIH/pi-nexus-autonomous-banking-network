import React, { useState, useEffect } from 'eact';
import { useWeb3React } from '@web3-react/core';
import { useContractLoader, useContractReader } from 'eth-hooks';
import { GoogleMap, LoadScript, Marker } from '@react-google-maps/api';

const SmartShipMap = () => {
  const { account, library } = useWeb3React();
  const contract = useContractLoader('SmartShip', library);
  const [shipments, setShipments] = useState([]);
  const [map, setMap] = useState(null);

  useEffect(() => {
    async function fetchData() {
      const shipments = await contract.methods.getShipments().call();
      setShipments(shipments);
    }
    fetchData();
  }, [account, library]);

  const handleMapLoad = (map) => {
    setMap(map);
  };

  return (
    <LoadScript googleMapsApiKey="YOUR_API_KEY">
      <GoogleMap
        center={{ lat: 37.7749, lng: -122.4194 }}
        zoom={12}
        onLoad={handleMapLoad}
      >
        {shipments.map((shipment, index) => (
          <Marker
            key={index}
            position={{ lat: shipment.latitude, lng: shipment.longitude }}
            title={shipment.id}
          />
        ))}
      </GoogleMap>
    </LoadScript>
  );
};

export default SmartShipMap;
