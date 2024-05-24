// models/PartnerMerchant.js
const mongoose = require('mongoose');

const partnerMerchantSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    required: true,
  },
  rewards: [
    {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Reward',
    },
  ],
});

module.exports = mongoose.model('PartnerMerchant', partnerMerchantSchema);
