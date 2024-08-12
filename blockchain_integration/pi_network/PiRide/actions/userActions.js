import { createAction } from 'redux-actions';
import { UserContract } from '../blockchain/smartContracts/UserContract';
import { Web3Provider } from '../providers/Web3Provider';
import { NotificationContext } from '../contexts/NotificationContext';

export const UPDATE_USER_PROFILE = 'UPDATE_USER_PROFILE';
export const UPDATE_USER_PROFILE_SUCCESS = 'UPDATE_USER_PROFILE_SUCCESS';
export const UPDATE_USER_PROFILE_FAILURE = 'UPDATE_USER_PROFILE_FAILURE';

export const GET_USER_PROFILE = 'GET_USER_PROFILE';
export const GET_USER_PROFILE_SUCCESS = 'GET_USER_PROFILE_SUCCESS';
export const GET_USER_PROFILE_FAILURE = 'GET_USER_PROFILE_FAILURE';

const updateUserProfile = createAction(UPDATE_USER_PROFILE, (name, email, phone, address) => ({
  name,
  email,
  phone,
  address,
}));

const getUserProfile = createAction(GET_USER_PROFILE);

export const updateUserProfileAsync = (name, email, phone, address) => async (dispatch) => {
  dispatch(updateUserProfile({ name, email, phone, address }));
  try {
    const web3Provider = new Web3Provider();
    const userContract = new UserContract(web3Provider);
    const txHash = await userContract.updateUserProfile(name, email, phone, address);
    dispatch({ type: UPDATE_USER_PROFILE_SUCCESS, txHash });
    NotificationContext.notify(`User profile updated successfully! Tx Hash: ${txHash}`);
  } catch (error) {
    dispatch({ type: UPDATE_USER_PROFILE_FAILURE, error });
    NotificationContext.notify(`Error updating user profile: ${error.message}`);
  }
};

export const getUserProfileAsync = () => async (dispatch) => {
  dispatch(getUserProfile());
  try {
    const web3Provider = new Web3Provider();
    const userContract = new UserContract(web3Provider);
    const userProfile = await userContract.getUserProfile();
    dispatch({ type: GET_USER_PROFILE_SUCCESS, userProfile });
  } catch (error) {
    dispatch({ type: GET_USER_PROFILE_FAILURE, error });
    NotificationContext.notify(`Error getting user profile: ${error.message}`);
  }
};
