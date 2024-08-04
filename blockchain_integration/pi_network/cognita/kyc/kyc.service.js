// kyc/kyc.service.js
import axios from 'axios';
import { PiNetworkApi } from './pi-network-api/pi-network-api';

const piNetworkApiUrl = 'https://api.pinetwork.io/v1';

class KycService {
  async verifyKyc(userId, userData) {
    try {
      const response = await axios.post(`${piNetworkApiUrl}/kyc/verify`, {
        user_id: userId,
        data: userData,
      });

      if (response.data.success) {
        // KYC verification successful
        return true;
      } else {
        // KYC verification failed
        return false;
      }
    } catch (error) {
      console.error(error);
      return false;
    }
  }
}

export default KycService;
