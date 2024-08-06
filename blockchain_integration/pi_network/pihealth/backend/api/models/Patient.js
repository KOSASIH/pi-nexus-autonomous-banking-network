const mongoose = require('mongoose');

const patientSchema = new mongoose.Schema({
  name: String,
  dateOfBirth: Date,
  address: String,
  phone: String,
});

const Patient = mongoose.model('Patient', patientSchema);

module.exports = Patient;
