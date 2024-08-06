export const SMART_CONTRACT_ADDRESSES = {
  HealthRecordContract: '0x...HealthRecordContractAddress...',
  MedicalBillingContract: '0x...MedicalBillingContractAddress...',
};

export const DATA_FORMATS = {
  healthRecord: {
    id: 'string',
    patientId: 'string',
    medicalHistory: 'array',
    allergies: 'array',
    medications: 'array',
  },
  medicalBilling: {
    id: 'string',
    patientId: 'string',
    billingDate: 'date',
    amount: 'number',
  },
};

export const API_ERROR_CODES = {
  invalidRequest: 400,
  unauthorized: 401,
  notFound: 404,
  internalServerError: 500,
};

export const IOT_DEVICE_TYPES = {
  bloodPressureMonitor: 'bloodPressureMonitor',
  glucoseMonitor: 'glucoseMonitor',
  heartRateMonitor: 'heartRateMonitor',
};
