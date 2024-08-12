import React, { useState, useEffect } from 'react';
import { useWeb3 } from '@pi-network/web3-react';
import { RideContract } from '../blockchain/smartContracts/RideContract';
import { GoogleMap, LoadScript } from '@react-google-maps/api';
import { useGeolocation } from 'react-use';
import { toast } from 'react-toastify';

const RideRequest = () => {
  const { account, library } = useWeb3();
  const [rideDetails, setRideDetails] = useState({
    pickupLocation: '',
    dropoffLocation: '',
    pickupTime: '',
    rideType: '',
  });
  const [userLocation, setUserLocation] = useState(null);
  const [rideOffers, setRideOffers] = useState([]);
  const [selectedRideOffer, setSelectedRideOffer] = useState(null);

  useEffect(() => {
    const rideContract = new RideContract(library);
    rideContract.getRideOffers(account).then((offers) => setRideOffers(offers));
  }, [account, library]);

  const handleRideRequest = async () => {
    const rideContract = new RideContract(library);
    await rideContract.requestRide(rideDetails, account);
    toast.success('Ride request sent successfully!');
  };

  const handleRideOfferSelection = (offer) => {
    setSelectedRideOffer(offer);
  };

  const handleRideAcceptance = async () => {
    const rideContract = new RideContract(library);
    await rideContract.acceptRide(selectedRideOffer, account);
    toast.success('Ride accepted successfully!');
  };

  const handleLocationChange = (location) => {
    setUserLocation(location);
  };

  return (
    <div>
      <h2>Ride Request</h2>
      <form>
        <label>Pickup Location:</label>
        <input
          type="text"
          value={rideDetails.pickupLocation}
          onChange={(e) => setRideDetails({ ...rideDetails, pickupLocation: e.target.value })}
        />
        <br />
        <label>Dropoff Location:</label>
        <input
          type="text"
          value={rideDetails.dropoffLocation}
          onChange={(e) => setRideDetails({ ...rideDetails, dropoffLocation: e.target.value })}
        />
        <br />
        <label>Pickup Time:</label>
        <input
          type="datetime-local"
          value={rideDetails.pickupTime}
          onChange={(e) => setRideDetails({ ...rideDetails, pickupTime: e.target.value })}
        />
        <br />
        <label>Ride Type:</label>
        <select
          value={rideDetails.rideType}
          onChange={(e) => setRideDetails({ ...rideDetails, rideType: e.target.value })}
        >
          <option value="Economy">Economy</option>
          <option value="Premium">Premium</option>
          <option value="Luxury">Luxury</option>
        </select>
        <br />
        <button onClick={handleRideRequest}>Request Ride</button>
      </form>
      <h2>Ride Offers</h2>
      <ul>
        {rideOffers.map((offer, index) => (
          <li key={index}>
            <span>
              {offer.driverName} - {offer.rideType} - {offer.price} Pi
            </span>
            <button onClick={() => handleRideOfferSelection(offer)}>Select</button>
          </li>
        ))}
      </ul>
      {selectedRideOffer && (
        <div>
          <h2>Selected Ride Offer</h2>
          <p>
            Driver: {selectedRideOffer.driverName}
            <br />
            Ride Type: {selectedRideOffer.rideType}
            <br />
            Price: {selectedRideOffer.price} Pi
          </p>
          <button onClick={handleRideAcceptance}>Accept Ride</button>
        </div>
      )}
      <LoadScript googleMapsApiKey="YOUR_GOOGLE_MAPS_API_KEY">
        <GoogleMap
          center={{ lat: 37.7749, lng: -122.4194 }}
          zoom={12}
          onClick={(event) => handleLocationChange(event.latLng)}
        >
          {userLocation && (
            <Marker position={userLocation} />
          )}
        </GoogleMap>
      </LoadScript>
    </div>
  );
};

export default RideRequest;
