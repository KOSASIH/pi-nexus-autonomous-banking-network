import mongoose from 'mongoose';
import { v4 as uuidv4 } from 'uuid';

const transactionSchema = new mongoose.Schema({
  _id: {
    type: String,
    required: true,
    unique: true,
    default: uuidv4,
  },
  piCoinId: {
    type: String,
    required: true,
    ref: 'PiCoin',
  },
  from: {
    type: String,
    required: true,
  },
  to: {
    type: String,
    required: true,
  },
  value: {
    type: Number,
    required: true,
  },
  gas: {
    type: Number,
    required: true,
  },
  gasPrice: {
    type: Number,
    required: true,
  },
  nonce: {
    type: Number,
    required: true,
  },
  txHash: {
    type: String,
    required: true,
  },
  blockNumber: {
    type: Number,
    required: true,
  },
  timestamp: {
    type: Date,
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

const Transaction = mongoose.model('Transaction', transactionSchema);

export default Transaction;
