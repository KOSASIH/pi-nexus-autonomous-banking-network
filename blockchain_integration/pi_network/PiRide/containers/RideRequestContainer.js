import React, { useState, useEffect, useContext } from 'react';
import { useWeb3 } from '@pi-network/web3-react';
import { RideContract } from '../blockchain/smartContracts/RideContract';
import { GoogleMapsAPI } from '../utils/GoogleMapsAPI';
import { RideRequestForm } from '../components/RideRequestForm';
import { RideRequestMap } from '../components/RideRequestMap';
import { RideRequestList } from '../components/RideRequestList';
import { NotificationContext } from '../contexts/NotificationContext';

const RideRequestContainer = () => {
  const { account, library } = useWeb3();
  const [rideRequests, setRideRequests] = useState([]);
  const [pickupLocation, setPickupLocation] = useState('');
  const [dropoffLocation, setDropoffLocation] = useState('');
  const [rideType, setRideType] = useState('');
  const [price, setPrice] = useState(0);
  const [loading, setLoading] = useState(false);
  const { notify } = useContext(NotificationContext);

  useEffect(() => {
    const rideContract = new RideContract(library);
    rideContract.getRideRequests(account).then((requests) => setRideRequests(requests));
  }, [account, library]);

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const rideContract = new RideContract(library);
      const txHash = await rideContract.createRideRequest(
        account,
        pickupLocation,
        dropoffLocation,
        rideType,
        price
      );
      notify(`Ride request created successfully! Tx Hash: ${txHash}`);
      setRideRequests([...rideRequests, { pickupLocation, dropoffLocation, rideType, price }]);
      setLoading(false);
    } catch (error) {
      notify(`Error creating ride request: ${error.message}`);
      setLoading(false);
    }
  };

  const handleMapClick = (location) => {
    setPickupLocation(location);
  };

  const handleDropoffChange = (location) => {
    setDropoffLocation(location);
  };

  const handleRideTypeChange = (type) => {
    setRideType(type);
  };

  const handlePriceChange = (price) => {
    setPrice(price);
  };

  return (
    <div>
      <h2>Ride Request Container</h2>
      <RideRequestForm
        onSubmit={handleFormSubmit}
        pickupLocation={pickupLocation}
        dropoffLocation={dropoffLocation}
        rideType={rideType}
        price={price}
        onChangePickupLocation={handleMapClick}
        onChangeDropoffLocation={handleDropoffChange}
        onChangeRideType={handleRideTypeChange}
        onChangePrice={handlePriceChange}
      />
      <RideRequestMap
        pickupLocation={pickupLocation}
        dropoffLocation={dropoffLocation}
        onClick={handleMapClick}
      />
      <RideRequestList rideRequests={rideRequests} />
      {loading && <p>Loading...</p>}
    </div>
  );
};

export default RideRequestContainer;
