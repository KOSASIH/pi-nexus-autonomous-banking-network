const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  address: String,
  name: String,
  email: String,
  phoneNumber: String,
  identityVerified: Boolean
});

module.exports = mongoose.model('User', userSchema);
