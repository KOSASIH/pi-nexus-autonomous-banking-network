import { model, Schema } from 'mongoose';
import { v4 as uuidv4 } from 'uuid';

const transactionSchema = new Schema({
  _id: { type: String, default: uuidv4 },
  accountId: { type: String, ref: 'Account' },
  amount: { type: Number, required: true },
  currency: { type: String, enum: ['USD', 'EUR', 'GBP'] },
  type: { type: String, enum: ['deposit', 'withdrawal'] },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

transactionSchema.virtual('account', {
  ref: 'Account',
  localField: 'accountId',
  foreignField: '_id'
});

export default model('Transaction', transactionSchema);
