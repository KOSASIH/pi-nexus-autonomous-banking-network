import axios from 'axios';

const AutonomousVehicleAPI = {
  fetchVehicleConfig: async () => {
    const response = await axios.get('/api/autonomous-vehicle/config');
    return response.data;
  },
  fetchMapConfig: async () => {
    const response = await axios.get('/api/map/config');
    return response.data;
  },
  fetchRoute: async () => {
    const response = await axios.get('/api/route');
    return response.data;
  },
  updateSensorData: async (sensorData) => {
    const response = await axios.post('/api/sensor-data', sensorData);
    return response.data;
  },
};

export default AutonomousVehicleAPI;
