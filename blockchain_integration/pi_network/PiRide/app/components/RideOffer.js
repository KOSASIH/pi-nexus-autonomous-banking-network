import React, { useState, useEffect } from 'react';
import { useWeb3 } from '@pi-network/web3-react';
import { RideContract } from '../blockchain/smartContracts/RideContract';
import { GoogleMap, LoadScript } from '@react-google-maps/api';
import { useGeolocation } from 'react-use';
import { toast } from 'react-toastify';

const RideOffer = () => {
  const { account, library } = useWeb3();
  const [rideDetails, setRideDetails] = useState({
    pickupLocation: '',
    dropoffLocation: '',
    pickupTime: '',
    rideType: '',
    price: 0,
  });
  const [userLocation, setUserLocation] = useState(null);

  useEffect(() => {
    const rideContract = new RideContract(library);
    rideContract.getRideRequests(account).then((requests) => {
      // Filter ride requests that match the driver's location and availability
      const filteredRequests = requests.filter((request) => {
        // Check if the request pickup location is within 5 miles of the driver's location
        const distance = calculateDistance(userLocation, request.pickupLocation);
        return distance <= 5;
      });
      setRideDetails(filteredRequests[0]);
    });
  }, [account, library, userLocation]);

  const handleRideOffer = async () => {
    const rideContract = new RideContract(library);
    await rideContract.offerRide(rideDetails, account);
    toast.success('Ride offer sent successfully!');
  };

  const handleLocationChange = (location) => {
    setUserLocation(location);
  };

  return (
    <div>
      <h2>Ride Offer</h2>
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
        <label>Price:</label>
        <input
          type="number"
          value={rideDetails.price}
          onChange={(e) => setRideDetails({ ...rideDetails, price: e.target.value })}
        />
        <br />
        <button onClick={handleRideOffer}>Offer Ride</button>
      </form>
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

export default RideOffer;
