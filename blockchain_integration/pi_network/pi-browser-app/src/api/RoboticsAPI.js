import axios from 'axios';

const RoboticsAPI = {
  fetchRobotConfig: async () => {
    const response = await axios.get('/api/robotics/config');
    return response.data;
  },
};

export default RoboticsAPI;
