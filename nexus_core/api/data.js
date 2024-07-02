import axios from 'axios';

const getData = async () => {
  try {
    const response = await axios.get('/api/data');
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export default getData;
