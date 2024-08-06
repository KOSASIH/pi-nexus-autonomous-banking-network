const mongoose = require('mongoose');

const medicalBillingSchema = new mongoose.Schema({
  patient: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient' },
  doctor: { type: mongoose.Schema.Types.ObjectId, ref: 'Doctor' },
  hospital: { type: mongoose.Schema.Types.ObjectId, ref: 'Hospital' },
  procedure: String,
  cost: Number,
});

const MedicalBilling = mongoose.model('MedicalBilling', medicalBillingSchema);

module.exports = MedicalBilling;
