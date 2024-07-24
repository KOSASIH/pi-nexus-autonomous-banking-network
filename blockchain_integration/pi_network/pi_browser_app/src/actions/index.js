import axios from 'axios';

export const trainModel = () => async (dispatch) => {
  try {
    const response = await axios.post('/api/train-model');
    dispatch({ type: 'TRAIN_MODEL_SUCCESS', payload: response.data });
  } catch (error) {
    dispatch({ type: 'TRAIN_MODEL_FAILURE', payload: error.message });
  }
};

export const predict = (inputData) => async (dispatch) => {
  try {
    const response = await axios.post('/api/predict', { inputData });
    dispatch({ type: 'PREDICT_SUCCESS', payload: response.data });
  } catch (error) {
    dispatch({ type: 'PREDICT_FAILURE', payload: error.message });
  }
};
