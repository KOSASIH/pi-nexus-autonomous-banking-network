const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const transactionSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: 'User' },
  accountId: { type: Schema.Types.ObjectId, ref: 'Account' },
  category: { type: String, enum: ['income', 'expense'] },
  amount: { type: Number, required: true },
  date: { type: Date, default: Date.now },
  description: { type: String }
});

module.exports = mongoose.model('Transaction', transactionSchema);
