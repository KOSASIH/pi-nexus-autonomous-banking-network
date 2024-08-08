import mongoose from 'mongoose';
import bcrypt from 'bcrypt';

const piCoinSchema = new mongoose.Schema({
  _id: {
    type: String,
    required: true,
    unique: true,
  },
  name: {
    type: String,
    required: true,
  },
  symbol: {
    type: String,
    required: true,
  },
  totalSupply: {
    type: Number,
    required: true,
  },
  circulatingSupply: {
    type: Number,
    required: true,
  },
  blockchain: {
    type: String,
    required: true,
  },
  contractAddress: {
    type: String,
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

piCoinSchema.pre('save', async function(next) {
  const piCoin = this;
  if (piCoin.isModified('contractAddress')) {
    const contractAddressHash = await bcrypt.hash(piCoin.contractAddress, 10);
    piCoin.contractAddressHash = contractAddressHash;
  }
  next();
});

piCoinSchema.methods.compareContractAddress = async function(contractAddress) {
  const piCoin = this;
  const isMatch = await bcrypt.compare(contractAddress, piCoin.contractAddressHash);
  return isMatch;
};

const PiCoin = mongoose.model('PiCoin', piCoinSchema);

export default PiCoin;
