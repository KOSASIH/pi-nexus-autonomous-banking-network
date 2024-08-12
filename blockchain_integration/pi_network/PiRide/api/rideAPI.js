import axios from 'axios';
import { API_URL } from '../constants';

const rideAPI = {
  createRideRequest: async (rideRequestData) => {
    try {
      const response = await axios.post(`${API_URL}/rides`, rideRequestData);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },

  getRideRequests: async () => {
    try {
      const response = await axios.get(`${API_URL}/rides`);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },

  createRideOffer: async (rideOfferData) => {
    try {
      const response = await axios.post(`${API_URL}/rides/offers`, rideOfferData);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },

  getRideOffers: async () => {
    try {
      const response = await axios.get(`${API_URL}/rides/offers`);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },

  acceptRideOffer: async (rideOfferId) => {
    try {
      const response = await axios.patch(`${API_URL}/rides/offers/${rideOfferId}/accept`);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },

  declineRideOffer: async (rideOfferId) => {
    try {
      const response = await axios.patch(`${API_URL}/rides/offers/${rideOfferId}/decline`);
      return response.data;
    } catch (error) {
      throw error.response.data;
    }
  },
};

export default rideAPI;
