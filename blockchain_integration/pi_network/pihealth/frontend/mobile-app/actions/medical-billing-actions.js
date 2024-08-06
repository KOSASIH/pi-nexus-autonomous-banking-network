import axios from 'axios';

export const GET_MEDICAL_BILLS_REQUEST = 'GET_MEDICAL_BILLS_REQUEST';
export const GET_MEDICAL_BILLS_SUCCESS = 'GET_MEDICAL_BILLS_SUCCESS';
export const GET_MEDICAL_BILLS_FAILURE = 'GET_MEDICAL_BILLS_FAILURE';

export const CREATE_MEDICAL_BILL_REQUEST = 'CREATE_MEDICAL_BILL_REQUEST';
export const CREATE_MEDICAL_BILL_SUCCESS = 'CREATE_MEDICAL_BILL_SUCCESS';
export const CREATE_MEDICAL_BILL_FAILURE = 'CREATE_MEDICAL_BILL_FAILURE';

export const UPDATE_MEDICAL_BILL_REQUEST = 'UPDATE_MEDICAL_BILL_REQUEST';
export const UPDATE_MEDICAL_BILL_SUCCESS = 'UPDATE_MEDICAL_BILL_SUCCESS';
export const UPDATE_MEDICAL_BILL_FAILURE = 'UPDATE_MEDICAL_BILL_FAILURE';

export const DELETE_MEDICAL_BILL_REQUEST = 'DELETE_MEDICAL_BILL_REQUEST';
export const DELETE_MEDICAL_BILL_SUCCESS = 'DELETE_MEDICAL_BILL_SUCCESS';
export const DELETE_MEDICAL_BILL_FAILURE = 'DELETE_MEDICAL_BILL_FAILURE';

export const getMedicalBills = () => async (dispatch) => {
  dispatch({ type: GET_MEDICAL_BILLS_REQUEST });
  try {
    const response = await axios.get('/api/medical-bills');
    dispatch({ type: GET_MEDICAL_BILLS_SUCCESS, payload: response.data });
  } catch (error) {
    dispatch({ type: GET_MEDICAL_BILLS_FAILURE, payload: error.message });
  }
};

export const createMedicalBill = (medicalBill) => async (dispatch) => {
  dispatch({ type: CREATE_MEDICAL_BILL_REQUEST });
  try {
    const response = await axios.post('/api/medical-bills', medicalBill);
    dispatch({ type: CREATE_MEDICAL_BILL_SUCCESS, payload: response.data });
  } catch (error) {
    dispatch({ type: CREATE_MEDICAL_BILL_FAILURE, payload: error.message });
  }
};

export const updateMedicalBill = (medicalBill) => async (dispatch) => {
  dispatch({ type: UPDATE_MEDICAL_BILL_REQUEST });
  try {
    const response = await axios.put(`/api/medical-bills/${medicalBill.id}`, medicalBill);
    dispatch({ type: UPDATE_MEDICAL_BILL_SUCCESS, payload: response.data });
  } catch (error) {
    dispatch({ type: UPDATE_MEDICAL_BILL_FAILURE, payload: error.message });
  }
};

export const deleteMedicalBill = (id) => async (dispatch) => {
  dispatch({ type: DELETE_MEDICAL_BILL_REQUEST });
  try {
    await axios.delete(`/api/medical-bills/${id}`);
    dispatch({ type: DELETE_MEDICAL_BILL_SUCCESS, payload: id });
  } catch (error) {
    dispatch({ type: DELETE_MEDICAL_BILL_FAILURE, payload: error.message });
  }
};
