const express = require('express');
const router = express.Router();
const HealthRecordController = require('../controllers/HealthRecordController');

// Create a new health record
router.post('/', HealthRecordController.createHealthRecord);

// Get a health record by ID
router.get('/:id', HealthRecordController.getHealthRecordById);

// Update a health record
router.put('/:id', HealthRecordController.updateHealthRecord);

// Delete a health record
router.delete('/:id', HealthRecordController.deleteHealthRecord);

// Get all health records for a patient
router.get('/patient/:patientId', HealthRecordController.getHealthRecordsByPatient);

// Get all health records for a doctor
router.get('/doctor/:doctorId', HealthRecordController.getHealthRecordsByDoctor);

// Get all health records for a hospital
router.get('/hospital/:hospitalId', HealthRecordController.getHealthRecordsByHospital);

module.exports = router;
