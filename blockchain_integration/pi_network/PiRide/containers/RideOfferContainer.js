import React, { useState, useEffect, useContext } from 'react';
import { useWeb3 } from '@pi-network/web3-react';
import { RideContract } from '../blockchain/smartContracts/RideContract';
import { GoogleMapsAPI } from '../utils/GoogleMapsAPI';
import { RideOfferForm } from '../components/RideOfferForm';
import { RideOfferMap } from '../components/RideOfferMap';
import { RideOfferList } from '../components/RideOfferList';
import { NotificationContext } from '../contexts/NotificationContext';

const RideOfferContainer = () => {
  const { account, library } = useWeb3();
  const [rideOffers, setRideOffers] = useState([]);
  const [pickupLocation, setPickupLocation] = useState('');
  const [dropoffLocation, setDropoffLocation] = useState('');
  const [rideType, setRideType] = useState('');
  const [price, setPrice] = useState(0);
  const [loading, setLoading] = useState(false);
  const { notify } = useContext(NotificationContext);

  useEffect(() => {
    const rideContract = new RideContract(library);
    rideContract.getRideOffers(account).then((offers) => setRideOffers(offers));
  }, [account, library]);

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const rideContract = new RideContract(library);
      const txHash = await rideContract.createRideOffer(
        account,
        pickupLocation,
        dropoffLocation,
        rideType,
        price
      );
      notify(`Ride offer created successfully! Tx Hash: ${txHash}`);
      setRideOffers([...rideOffers, { pickupLocation, dropoffLocation, rideType, price }]);
      setLoading(false);
    } catch (error) {
      notify(`Error creating ride offer: ${error.message}`);
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
      <h2>Ride Offer Container</h2>
      <RideOfferForm
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
      <RideOfferMap
        pickupLocation={pickupLocation}
        dropoffLocation={dropoffLocation}
        onClick={handleMapClick}
      />
      <RideOfferList rideOffers={rideOffers} />
      {loading && <p>Loading...</p>}
    </div>
  );
};

export default RideOfferContainer;
