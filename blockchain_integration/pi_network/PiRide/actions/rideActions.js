import { createAction } from 'redux-actions';
import { RideContract } from '../blockchain/smartContracts/RideContract';
import { Web3Provider } from '../providers/Web3Provider';
import { NotificationContext } from '../contexts/NotificationContext';

export const CREATE_RIDE_REQUEST = 'CREATE_RIDE_REQUEST';
export const CREATE_RIDE_REQUEST_SUCCESS = 'CREATE_RIDE_REQUEST_SUCCESS';
export const CREATE_RIDE_REQUEST_FAILURE = 'CREATE_RIDE_REQUEST_FAILURE';

export const GET_RIDE_REQUESTS = 'GET_RIDE_REQUESTS';
export const GET_RIDE_REQUESTS_SUCCESS = 'GET_RIDE_REQUESTS_SUCCESS';
export const GET_RIDE_REQUESTS_FAILURE = 'GET_RIDE_REQUESTS_FAILURE';

export const CREATE_RIDE_OFFER = 'CREATE_RIDE_OFFER';
export const CREATE_RIDE_OFFER_SUCCESS = 'CREATE_RIDE_OFFER_SUCCESS';
export const CREATE_RIDE_OFFER_FAILURE = 'CREATE_RIDE_OFFER_FAILURE';

export const GET_RIDE_OFFERS = 'GET_RIDE_OFFERS';
export const GET_RIDE_OFFERS_SUCCESS = 'GET_RIDE_OFFERS_SUCCESS';
export const GET_RIDE_OFFERS_FAILURE = 'GET_RIDE_OFFERS_FAILURE';

const createRideRequest = createAction(CREATE_RIDE_REQUEST, (pickupLocation, dropoffLocation, rideType, price) => ({
  pickupLocation,
  dropoffLocation,
  rideType,
  price,
}));

const getRideRequests = createAction(GET_RIDE_REQUESTS);

const createRideOffer = createAction(CREATE_RIDE_OFFER, (pickupLocation, dropoffLocation, rideType, price) => ({
  pickupLocation,
  dropoffLocation,
  rideType,
  price,
}));

const getRideOffers = createAction(GET_RIDE_OFFERS);

export const createRideRequestAsync = (pickupLocation, dropoffLocation, rideType, price) => async (dispatch) => {
  dispatch(createRideRequest({ pickupLocation, dropoffLocation, rideType, price }));
  try {
    const web3Provider = new Web3Provider();
    const rideContract = new RideContract(web3Provider);
    const txHash = await rideContract.createRideRequest(pickupLocation, dropoffLocation, rideType, price);
    dispatch({ type: CREATE_RIDE_REQUEST_SUCCESS, txHash });
    NotificationContext.notify(`Ride request created successfully! Tx Hash: ${txHash}`);
  } catch (error) {
    dispatch({ type: CREATE_RIDE_REQUEST_FAILURE, error });
    NotificationContext.notify(`Error creating ride request: ${error.message}`);
  }
};

export const getRideRequestsAsync = () => async (dispatch) => {
  dispatch(getRideRequests());
  try {
    const web3Provider = new Web3Provider();
    const rideContract = new RideContract(web3Provider);
    const rideRequests = await rideContract.getRideRequests();
    dispatch({ type: GET_RIDE_REQUESTS_SUCCESS, rideRequests });
  } catch (error) {
    dispatch({ type: GET_RIDE_REQUESTS_FAILURE, error });
    NotificationContext.notify(`Error getting ride requests: ${error.message}`);
  }
};

export const createRideOfferAsync = (pickupLocation, dropoffLocation, rideType, price) => async (dispatch) => {
  dispatch(createRideOffer({ pickupLocation, dropoffLocation, rideType, price }));
  try {
    const web3Provider = new Web3Provider();
    const rideContract = new RideContract(web3Provider);
    const txHash = await rideContract.createRideOffer(pickupLocation, dropoffLocation, rideType, price);
    dispatch({ type: CREATE_RIDE_OFFER_SUCCESS, txHash });
    NotificationContext.notify(`Ride offer created successfully! Tx Hash: ${txHash}`);
  } catch (error) {
    dispatch({ type: CREATE_RIDE_OFFER_FAILURE, error });
    NotificationContext.notify(`Error creating ride offer: ${error.message}`);
  }
};

export const getRideOffersAsync = () => async (dispatch) => {
  dispatch(getRideOffers());
  try {
    const web3Provider = new Web3Provider();
    const rideContract = new RideContract(web3Provider);
    const rideOffers = await rideContract.getRideOffers();
    dispatch({ type: GET_RIDE_OFFERS_SUCCESS, rideOffers });
  } catch (error) {
    dispatch({ type: GET_RIDE_OFFERS_FAILURE, error });
    NotificationContext.notify(`Error getting ride offers: ${error.message}`);
  }
};
