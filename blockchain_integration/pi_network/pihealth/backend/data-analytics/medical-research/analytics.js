const express = require('express');
const router = express.Router();
const { MedicalResearch } = require('../models/MedicalResearch');
const { Patient } = require('../models/Patient');
const { Doctor } = require('../models/Doctor');
const { Hospital } = require('../models/Hospital');
const { MedicalRecord } = require('../models/MedicalRecord');
const { MedicalBilling } = require('../models/MedicalBilling');
const { MLModel } = require('../models/MLModel');
const { predictDisease } = require('../utils/predictDisease');
const { predictTreatmentOutcome } = require('../utils/predictTreatmentOutcome');
const { predictPatientRisk } = require('../utils/predictPatientRisk');

// Get analytics for medical research
router.get('/analytics', async (req, res) => {
  try {
    const patients = await Patient.find();
    const doctors = await Doctor.find();
    const hospitals = await Hospital.find();
    const medicalRecords = await MedicalRecord.find();
    const medicalBillings = await MedicalBilling.find();

    const patientCount = patients.length;
    const doctorCount = doctors.length;
    const hospitalCount = hospitals.length;
    const medicalRecordCount = medicalRecords.length;
    const medicalBillingCount = medicalBillings.length;

    const diseasePredictions = await predictDisease(medicalRecords);
    const treatmentOutcomePredictions = await predictTreatmentOutcome(medicalRecords);
    const patientRiskPredictions = await predictPatientRisk(medicalRecords);

    const analytics = {
      patientCount,
      doctorCount,
      hospitalCount,
      medicalRecordCount,
      medicalBillingCount,
      diseasePredictions,
      treatmentOutcomePredictions,
      patientRiskPredictions,
    };

    res.json(analytics);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting analytics' });
  }
});

// Get analytics for a specific disease
router.get('/analytics/disease/:disease', async (req, res) => {
  try {
    const disease = req.params.disease;
    const medicalRecords = await MedicalRecord.find({ disease });

    const diseasePredictions = await predictDisease(medicalRecords);
    const treatmentOutcomePredictions = await predictTreatmentOutcome(medicalRecords);
    const patientRiskPredictions = await predictPatientRisk(medicalRecords);

    const analytics = {
      diseasePredictions,
      treatmentOutcomePredictions,
      patientRiskPredictions,
    };

    res.json(analytics);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting analytics for disease' });
  }
});

// Get analytics for a specific treatment
router.get('/analytics/treatment/:treatment', async (req, res) => {
  try {
    const treatment = req.params.treatment;
    const medicalRecords = await MedicalRecord.find({ treatment });

    const treatmentOutcomePredictions = await predictTreatmentOutcome(medicalRecords);
    const patientRiskPredictions = await predictPatientRisk(medicalRecords);

    const analytics = {
      treatmentOutcomePredictions,
      patientRiskPredictions,
    };

    res.json(analytics);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting analytics for treatment' });
  }
});

// Train machine learning model
router.post('/train-model', async (req, res) => {
  try {
    const mlModel = new MLModel();
    await mlModel.train();
    res.json({ message: 'Model trained successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error training model' });
  }
});

module.exports = router;
