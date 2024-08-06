const express = require('express');
const router = express.Router();
const MedicalBillingController = require('../controllers/MedicalBillingController');

// Create a new medical billing record
router.post('/', MedicalBillingController.createMedicalBillingRecord);

// Get a medical billing record by ID
router.get('/:id', MedicalBillingController.getMedicalBillingRecordById);

// Update a medical billing record
router.put('/:id', MedicalBillingController.updateMedicalBillingRecord);

// Delete a medical billing record
router.delete('/:id', MedicalBillingController.deleteMedicalBillingRecord);

// Get all medical billing records for a patient
router.get('/patient/:patientId', MedicalBillingController.getMedicalBillingRecordsByPatient);

// Get all medical billing records for a doctor
router.get('/doctor/:doctorId', MedicalBillingController.getMedicalBillingRecordsByDoctor);

// Get all medical billing records for a hospital
router.get('/hospital/:hospitalId', MedicalBillingController.getMedicalBillingRecordsByHospital);

module.exports = router;
