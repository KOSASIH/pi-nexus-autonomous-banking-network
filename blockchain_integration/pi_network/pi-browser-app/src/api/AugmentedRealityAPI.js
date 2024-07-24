import axios from 'axios';

const AugmentedRealityAPI = {
  getMarker: async () => {
    const response = await axios.get('/api/ar/marker');
    return response.data;
  },
};

export default AugmentedRealityAPI;
