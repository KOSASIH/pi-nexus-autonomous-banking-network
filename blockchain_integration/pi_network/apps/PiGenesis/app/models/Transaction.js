import mongoose, { Document, Model, Schema } from 'ongoose';

interface Transaction {
  from: string;
  to: string;
  amount: number;
  timestamp: Date;
}

const transactionSchema = new Schema<Transaction>({
  from: { type: String, required: true },
  to: { type: String, required: true },
  amount: { type: Number, required: true },
  timestamp: { type: Date, required: true },
});

const Transaction: Model<Transaction> = mongoose.model('Transaction', transactionSchema);

export default Transaction;
