const mongoose = require("mongoose");

const transferSchema = new mongoose.Schema({
  sender_id: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  receiver_account_id: { type: String, required: true },
  amount: { type: Number, required: true },
  transaction_hash: { type: String },
});

const Transfer = mongoose.model("Transfer", transferSchema);

module.exports = Transfer;
