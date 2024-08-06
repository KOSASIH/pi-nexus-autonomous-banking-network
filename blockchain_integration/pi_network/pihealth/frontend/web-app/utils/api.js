import axios from 'axios';

const API_URL = 'https://api.pihealth.io';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealthRecords = () => api.get('/health-records');
export const getMedicalBills = () => api.get('/medical-bills');
export const createHealthRecord = (data) => api.post('/health-records', data);
export const createMedicalBill = (data) => api.post('/medical-bills', data);
export const updateHealthRecord = (id, data) => api.patch(`/health-records/${id}`, data);
export const updateMedicalBill = (id, data) => api.patch(`/medical-bills/${id}`, data);
export const deleteHealthRecord = (id) => api.delete(`/health-records/${id}`);
export const deleteMedicalBill = (id) => api.delete(`/medical-bills/${id}`);
