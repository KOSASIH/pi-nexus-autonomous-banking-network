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
      notify
