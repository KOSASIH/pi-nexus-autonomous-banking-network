const mongoose = require('mongoose');

const doctorSchema = new mongoose.Schema({
  name: String,
  specialty: String,
  hospital: { type: mongoose.Schema.Types.ObjectId, ref: 'Hospital' },
});

const Doctor = mongoose.model('Doctor', doctorSchema);

module.exports = Doctor;
