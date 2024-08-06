const express = require('express');
const router = express.Router();
const MedicalBilling = require('../models/MedicalBilling');
const Patient = require('../models/Patient');
const Doctor = require('../models/Doctor');
const Hospital = require('../models/Hospital');
const { validateMedicalBilling } = require('../utils/validation');

// Create a new medical billing record
router.post('/', async (req, res) => {
  try {
    const { patientId, doctorId, hospitalId, procedure, cost } = req.body;
    const patient = await Patient.findById(patientId);
    const doctor = await Doctor.findById(doctorId);
    const hospital = await Hospital.findById(hospitalId);

    if (!patient || !doctor || !hospital) {
      return res.status(404).json({ message: 'Patient, doctor, or hospital not found' });
    }

    const medicalBilling = new MedicalBilling({
      patient: patient._id,
      doctor: doctor._id,
      hospital: hospital._id,
      procedure,
      cost,
    });

    await medicalBilling.save();
    res.json({ message: 'Medical billing record created successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error creating medical billing record' });
  }
});

// Get a medical billing record by ID
router.get('/:id', async (req, res) => {
  try {
    const medicalBilling = await MedicalBilling.findById(req.params.id);
    if (!medicalBilling) {
      return res.status(404).json({ message: 'Medical billing record not found' });
    }
    res.json(medicalBilling);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting medical billing record' });
  }
});

// Update a medical billing record
router.put('/:id', async (req, res) => {
  try {
    const medicalBilling = await MedicalBilling.findById(req.params.id);
    if (!medicalBilling) {
      return res.status(404).json({ message: 'Medical billing record not found' });
    }

    const { procedure, cost } = req.body;
    medicalBilling.procedure = procedure;
    medicalBilling.cost = cost;

    await medicalBilling.save();
    res.json({ message: 'Medical billing record updated successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error updating medical billing record' });
  }
});

// Delete a medical billing record
router.delete('/:id', async (req, res) => {
  try {
    const medicalBilling = await MedicalBilling.findById(req.params.id);
    if (!medicalBilling) {
      return res.status(404).json({ message: 'Medical billing record not found' });
    }

    await medicalBilling.remove();
    res.json({ message: 'Medical billing record deleted successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error deleting medical billing record' });
  }
});

// Get all medical billing records for a patient
router.get('/patient/:patientId', async (req, res) => {
  try {
    const medicalBillings = await MedicalBilling.find({ patient: req.params.patientId });
    res.json(medicalBillings);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting medical billing records' });
  }
});

// Get all medical billing records for a doctor
router.get('/doctor/:doctorId', async (req, res) => {
  try {
    const medicalBillings = await MedicalBilling.find({ doctor: req.params.doctorId });
    res.json(medicalBillings);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting medical billing records' });
  }
});

// Get all medical billing records for a hospital
router.get('/hospital/:hospitalId', async (req, res) => {
  try {
    const medicalBillings = await MedicalBilling.find({ hospital: req.params.hospitalId });
    res.json(medicalBillings);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting medical billing records' });
  }
});

module.exports = router;
