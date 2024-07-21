const mongoose = require('mongoose');

const identityVerificationSchema = new mongoose.Schema({
  userAddress: String,
  userData: String,
  verified: Boolean,
  timestamp: Date
});

module.exports = mongoose.model('IdentityVerification', identityVerificationSchema);
