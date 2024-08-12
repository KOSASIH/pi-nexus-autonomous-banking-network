import React, { useState, useEffect } from 'react';
import { useWeb3 } from '@pi-network/web3-react';
import { RideContract } from '../blockchain/smartContracts/RideContract';
import { calculateReputation } from '../reputationAlgorithm';

const UserProfile = () => {
  const { account, library } = useWeb3();
  const [user, setUser] = useState({});
  const [reputationScore, setReputationScore] = useState(0);
  const [rides, setRides] = useState([]);
  const [ratings, setRatings] = useState([]);

  useEffect(() => {
    const rideContract = new RideContract(library);
    rideContract.getUserData(account).then((data) => setUser(data));
  }, [account, library]);

  useEffect(() => {
    const rideContract = new RideContract(library);
    rideContract.getUserRides(account).then((rides) => setRides(rides));
  }, [account, library]);

  useEffect(() => {
    const rideContract = new RideContract(library);
    rideContract.getUserRatings(account).then((ratings) => setRatings(ratings));
  }, [account, library]);

  useEffect(() => {
    const ratingsArray = ratings.map((rating) => rating.rating);
    const reputationScore = calculateReputation(ratingsArray);
    setReputationScore(reputationScore);
  }, [ratings]);

  return (
    <div>
      <h2>User Profile</h2>
      <p>Name: {user.name}</p>
      <p>Email: {user.email}</p>
      <p>Phone: {user.phone}</p>
      <p>Address: {user.address}</p>
      <p>Reputation Score: {reputationScore}</p>
      <p>Number of Rides: {rides.length}</p>
      <p>Number of Ratings: {ratings.length}</p>
      <p>Total Earnings: {user.totalEarnings} Pi</p>
      <h3>Ride History</h3>
      <ul>
        {rides.map((ride, index) => (
          <li key={index}>
            <p>Pickup Location: {ride.pickupLocation}</p>
            <p>Dropoff Location: {ride.dropoffLocation}</p>
            <p>Pickup Time: {ride.pickupTime}</p>
            <p>Ride Type: {ride.rideType}</p>
            <p>Price: {ride.price} Pi</p>
          </li>
        ))}
      </ul>
      <h3>Ratings</h3>
      <ul>
        {ratings.map((rating, index) => (
          <li key={index}>
            <p>Rating: {rating.rating}</p>
            <p>Comment: {rating.comment}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default UserProfile;
