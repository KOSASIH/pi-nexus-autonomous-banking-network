import mongoose from 'mongoose';

const dataSchema = new mongoose.Schema({
  label: String,
  value: Number,
});

const Data = mongoose.model('Data', dataSchema);

export default Data;
