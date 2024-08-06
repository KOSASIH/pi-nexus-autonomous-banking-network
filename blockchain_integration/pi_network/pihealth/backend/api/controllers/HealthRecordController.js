const express = require('express');
const router = express.Router();
const HealthRecord = require('../models/HealthRecord');
const Patient = require('../models/Patient');
const Doctor = require('../models/Doctor');
const Hospital = require('../models/Hospital');
const { validateHealthRecord } = require('../utils/validation');

// Create a new health record
router.post('/', async (req, res) => {
  try {
    const { patientId, doctorId, hospitalId, medicalHistory, allergies, medications } = req.body;
    const patient = await Patient.findById(patientId);
    const doctor = await Doctor.findById(doctorId);
    const hospital = await Hospital.findById(hospitalId);

    if (!patient || !doctor || !hospital) {
      return res.status(404).json({ message: 'Patient, doctor, or hospital not found' });
    }

    const healthRecord = new HealthRecord({
      patient: patient._id,
      doctor: doctor._id,
      hospital: hospital._id,
      medicalHistory,
      allergies,
      medications,
    });

    await healthRecord.save();
    res.json({ message: 'Health record created successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error creating health record' });
  }
});

// Get a health record by ID
router.get('/:id', async (req, res) => {
  try {
    const healthRecord = await HealthRecord.findById(req.params.id);
    if (!healthRecord) {
      return res.status(404).json({ message: 'Health record not found' });
    }
    res.json(healthRecord);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting health record' });
  }
});

// Update a health record
router.put('/:id', async (req, res) => {
  try {
    const healthRecord = await HealthRecord.findById(req.params.id);
    if (!healthRecord) {
      return res.status(404).json({ message: 'Health record not found' });
    }

    const { medicalHistory, allergies, medications } = req.body;
    healthRecord.medicalHistory = medicalHistory;
    healthRecord.allergies = allergies;
    healthRecord.medications = medications;

    await healthRecord.save();
    res.json({ message: 'Health record updated successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error updating health record' });
  }
});

// Delete a health record
router.delete('/:id', async (req, res) => {
  try {
    const healthRecord = await HealthRecord.findById(req.params.id);
    if (!healthRecord) {
      return res.status(404).json({ message: 'Health record not found' });
    }

    await healthRecord.remove();
    res.json({ message: 'Health record deleted successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error deleting health record' });
  }
});

module.exports = router;
