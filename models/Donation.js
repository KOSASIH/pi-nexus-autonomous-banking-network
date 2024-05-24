// models/Donation.js
const mongoose = require('mongoose');

const donationSchema = new mongoose.Schema({
  transaction: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Transaction',
  },
  amount: {
    type: Number,
    required: true,
  },
  cause: {
    type: String,
    enum: ['environmental', 'social', 'other'],
    required: true,
  },
});

module.exports = mongoose.model('Donation', donationSchema);
