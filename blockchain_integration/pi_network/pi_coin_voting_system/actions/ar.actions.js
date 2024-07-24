import axios from 'axios';

export const getARDashboardData = () => {
  return (dispatch) => {
    dispatch({ type: 'AR_DATA_LOADING' });
    axios
      .get('https://example.com/ar-dashboard-data')
      .then((response) => {
        const data = response.data;
        dispatch({ type: 'AR_DATA_RECEIVED', data });
      })
      .catch((error) => {
        dispatch({ type: 'AR_DATA_ERROR', error });
      });
  };
};

export const getARModel = (id) => {
  return (dispatch) => {
    dispatch({ type: 'AR_MODEL_LOADING' });
    axios
      .get(`https://example.com/ar-models/${id}`)
      .then((response) => {
        const model = response.data;
        dispatch({ type: 'AR_MODEL_RECEIVED', model });
      })
      .catch((error) => {
        dispatch({ type: 'AR_MODEL_ERROR', error });
      });
  };
};

export const uploadARModel = (model) => {
  return (dispatch) => {
    dispatch({ type: 'AR_MODEL_UPLOADING' });
    axios
      .post('https://example.com/ar-models', model)
      .then((response) => {
        const model = response.data;
        dispatch({ type: 'AR_MODEL_UPLOADED', model });
      })
      .catch((error) => {
        dispatch({ type: 'AR_MODEL_UPLOAD_ERROR', error });
      });
  };
};
