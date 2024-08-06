import mongoose from 'mongoose';

const policySchema = new mongoose.Schema({
  firstName: { type: String, required: true },
  lastName: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  policyType: { type: String, required: true },
  policyId: { type: String, required: true, unique: true },
});

const Policy = mongoose.model('Policy', policySchema);

export { Policy };
