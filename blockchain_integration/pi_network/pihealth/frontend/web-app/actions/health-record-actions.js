import { GET_HEALTH_RECORDS, GET_HEALTH_RECORD, CREATE_HEALTH_RECORD, UPDATE_HEALTH_RECORD, DELETE_HEALTH_RECORD } from './health-record-queries';
import { GET_MEDICAL_BILLS } from '../medical-billing/medical-billing-queries';

export const getHealthRecords = () => ({
  type: 'GET_HEALTH_RECORDS',
  payload: {
    query: GET_HEALTH_RECORDS,
    update: (cache, { data: { healthRecords } }) => {
      cache.writeQuery({
        query: GET_HEALTH_RECORDS,
        data: { healthRecords },
      });
    },
  },
});

export const getHealthRecord = (id) => ({
  type: 'GET_HEALTH_RECORD',
  payload: {
    query: GET_HEALTH_RECORD,
    variables: { id },
    update: (cache, { data: { healthRecord } }) => {
      cache.writeQuery({
        query: GET_HEALTH_RECORD,
        variables: { id },
        data: { healthRecord },
      });
    },
  },
});

export const createHealthRecord = (healthRecord) => ({
  type: 'CREATE_HEALTH_RECORD',
  payload: {
    query: CREATE_HEALTH_RECORD,
    variables: { healthRecord },
    update: (cache, { data: { createHealthRecord } }) => {
      const { healthRecords } = cache.readQuery({ query: GET_HEALTH_RECORDS });
      cache.writeQuery({
        query: GET_HEALTH_RECORDS,
        data: { healthRecords: healthRecords.concat(createHealthRecord) },
      });
    },
  },
});

export const updateHealthRecord = (id, healthRecord) => ({
  type: 'UPDATE_HEALTH_RECORD',
  payload: {
    query: UPDATE_HEALTH_RECORD,
    variables: { id, healthRecord },
    update: (cache, { data: { updateHealthRecord } }) => {
      const { healthRecords } = cache.readQuery({ query: GET_HEALTH_RECORDS });
      const index = healthRecords.findIndex((record) => record.id === id);
      healthRecords[index] = updateHealthRecord;
      cache.writeQuery({
        query: GET_HEALTH_RECORDS,
        data: { healthRecords },
      });
    },
  },
});

export const deleteHealthRecord = (id) => ({
  type: 'DELETE_HEALTH_RECORD',
  payload: {
    query: DELETE_HEALTH_RECORD,
    variables: { id },
    update: (cache, { data: { deleteHealthRecord } }) => {
      const { healthRecords } = cache.readQuery({ query: GET_HEALTH_RECORDS });
      cache.writeQuery({
        query: GET_HEALTH_RECORDS,
        data: { healthRecords: healthRecords.filter((record) => record.id !== id) },
      });
    },
  },
});

export const getHealthRecordsWithMedicalBills = () => (dispatch) => {
  dispatch(getHealthRecords());
  dispatch(getMedicalBills());
};

export const getHealthRecordWithMedicalBills = (id) => (dispatch) => {
  dispatch(getHealthRecord(id));
  dispatch(getMedicalBills());
};
