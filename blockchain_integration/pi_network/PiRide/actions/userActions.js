import { createAction } from 'redux-actions';
import { UserContract } from '../blockchain/smartContracts/UserContract';
import { Web3Provider } from '../providers/Web3Provider';
import { NotificationContext } from '../contexts/NotificationContext';

export const UPDATE_USER_PROFILE = 'UPDATE_USER_PROFILE';
export const UPDATE_USER_PROFILE_SUCCESS = 'UPDATE
