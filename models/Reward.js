// models/Reward.js
const mongoose = require("mongoose");

const rewardSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    required: true,
  },
  pointsRequired: {
    type: Number,
    required: true,
  },
  discount: {
    type: Number,
    required: true,
  },
  partnerMerchant: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "PartnerMerchant",
  },
});

module.exports = mongoose.model("Reward", rewardSchema);
