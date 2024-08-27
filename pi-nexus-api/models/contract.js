import mongoose from 'mongoose';

const contractSchema = new mongoose.Schema({
  address: {
    type: String,
    required: true,
    unique: true
  },
  bytecode: {
    type: String,
    required: true
  },
  abi: {
    type: String,
    required: true
  },
  deployed: {
    type: Boolean,
    default: false
  }
});

const Contract = mongoose.model('Contract', contractSchema);

export default Contract;
