const mongoose = require('mongoose');

const medicalBillingSchema = new mongoose.Schema({
  patientName: { type: String, required: true },
  billingDate: { type: Date, required: true },
  amount: { type: Number, required: true },
});

const MedicalBilling = mongoose.model('MedicalBilling', medicalBillingSchema);

module.exports = MedicalBilling;
