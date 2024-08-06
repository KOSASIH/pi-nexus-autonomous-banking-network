import axios from 'axios';

export const GET_HEALTH_RECORDS_REQUEST = 'GET_HEALTH_RECORDS_REQUEST';
export const GET_HEALTH_RECORDS_SUCCESS = 'GET_HEALTH_RECORDS_SUCCESS';
export const GET_HEALTH_RECORDS_FAILURE = 'GET_HEALTH_RECORDS_FAILURE';

export const CREATE_HEALTH_RECORD_REQUEST = 'CREATE_HEALTH_RECORD_REQUEST';
export const CREATE_HEALTH_RECORD_SUCCESS = 'CREATE_HEALTH_RECORD_SUCCESS';
export const CREATE_HEALTH_RECORD_FAILURE = 'CREATE_HEALTH_RECORD_FAILURE';

export const UPDATE_HEALTH_RECORD_REQUEST = 'UPDATE_HEALTH_RECORD_REQUEST';
export const UPDATE_HEALTH_RECORD_SUCCESS = 'UPDATE_HEALTH_RECORD_SUCCESS';
export const UPDATE_HEALTH_RECORD_FAILURE = 'UPDATE_HEALTH_RECORD_FAILURE';

export const DELETE_HEALTH_RECORD_REQUEST = 'DELETE_HEALTH_RECORD_REQUEST';
export const DELETE_HEALTH_RECORD_SUCCESS = 'DELETE_HEALTH_RECORD_SUCCESS';
export const DELETE_HEALTH_RECORD_FAILURE = 'DELETE_HEALTH_RECORD_FAILURE';

export const getHealthRecords = () => async (dispatch) => {
  dispatch({ type: GET_HEALTH_RECORDS_REQUEST });
  try {
    const response = await axios.get('/api/health-records');
    dispatch({ type: GET_HEALTH_RECORDS_SUCCESS, payload: response.data });
  } catch (error) {
    dispatch({ type: GET_HEALTH_RECORDS_FAILURE, payload: error.message });
  }
};

export const createHealthRecord = (healthRecord) => async (dispatch) => {
  dispatch({ type: CREATE_HEALTH_RECORD_REQUEST });
  try {
    const response = await axios.post('/api/health-records', healthRecord);
    dispatch({ type: CREATE_HEALTH_RECORD_SUCCESS, payload: response.data });
  } catch (error) {
    dispatch({ type: CREATE_HEALTH_RECORD_FAILURE, payload: error.message });
  }
};

export const updateHealthRecord = (healthRecord) => async (dispatch) => {
  dispatch({ type: UPDATE_HEALTH_RECORD_REQUEST });
  try {
    const response = await axios.put(`/api/health-records/${healthRecord.id}`, healthRecord);
    dispatch({ type: UPDATE_HEALTH_RECORD_SUCCESS, payload: response.data });
  } catch (error) {
    dispatch({ type: UPDATE_HEALTH_RECORD_FAILURE, payload: error.message });
  }
};

export const deleteHealthRecord = (id) => async (dispatch) => {
  dispatch({ type: DELETE_HEALTH_RECORD_REQUEST });
  try {
    await axios.delete(`/api/health-records/${id}`);
    dispatch({ type: DELETE_HEALTH_RECORD_SUCCESS, payload: id });
  } catch (error) {
    dispatch({ type: DELETE_HEALTH_RECORD_FAILURE, payload: error.message });
  }
};
