import { GET_MEDICAL_BILLS, GET_MEDICAL_BILL, CREATE_MEDICAL_BILL, UPDATE_MEDICAL_BILL, DELETE_MEDICAL_BILL } from './medical-billing-queries';
import { GET_HEALTH_RECORDS } from '../health-record/health-record-queries';

export const getMedicalBills = () => ({
  type: 'GET_MEDICAL_BILLS',
  payload: {
    query: GET_MEDICAL_BILLS,
    update: (cache, { data: { medicalBills } }) => {
      cache.writeQuery({
        query: GET_MEDICAL_BILLS,
        data: { medicalBills },
      });
    },
  },
});

export const getMedicalBill = (id) => ({
  type: 'GET_MEDICAL_BILL',
  payload: {
    query: GET_MEDICAL_BILL,
    variables: { id },
    update: (cache, { data: { medicalBill } }) => {
      cache.writeQuery({
        query: GET_MEDICAL_BILL,
        variables: { id },
        data: { medicalBill },
      });
    },
  },
});

export const createMedicalBill = (medicalBill) => ({
  type: 'CREATE_MEDICAL_BILL',
  payload: {
    query: CREATE_MEDICAL_BILL,
    variables: { medicalBill },
    update: (cache, { data: { createMedicalBill } }) => {
      const { medicalBills } = cache.readQuery({ query: GET_MEDICAL_BILLS });
      cache.writeQuery({
        query: GET_MEDICAL_BILLS,
        data: { medicalBills: medicalBills.concat(createMedicalBill) },
      });
    },
  },
});

export const updateMedicalBill = (id, medicalBill) => ({
  type: 'UPDATE_MEDICAL_BILL',
  payload: {
    query: UPDATE_MEDICAL_BILL,
    variables: { id, medicalBill },
    update: (cache, { data: { updateMedicalBill } }) => {
      const { medicalBills } = cache.readQuery({ query: GET_MEDICAL_BILLS });
      const index = medicalBills.findIndex((bill) => bill.id === id);
      medicalBills[index] = updateMedicalBill;
      cache.writeQuery({
        query: GET_MEDICAL_BILLS,
        data: { medicalBills },
      });
    },
  },
});

export const deleteMedicalBill = (id) => ({
  type: 'DELETE_MEDICAL_BILL',
  payload: {
    query: DELETE_MEDICAL_BILL,
    variables: { id },
    update: (cache, { data: { deleteMedicalBill } }) => {
      const { medicalBills } = cache.readQuery({ query: GET_MEDICAL_BILLS });
      cache.writeQuery({
        query: GET_MEDICAL_BILLS,
        data: { medicalBills: medicalBills.filter((bill) => bill.id !== id) },
      });
    },
  },
});

export const getMedicalBillsWithHealthRecords = () => (dispatch) => {
  dispatch(getMedicalBills());
  dispatch(getHealthRecords());
};

export const getMedicalBillWithHealthRecords = (id) => (dispatch) => {
  dispatch(getMedicalBill(id));
  dispatch(getHealthRecords());
};
