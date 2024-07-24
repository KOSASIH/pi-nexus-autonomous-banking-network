import mongoose from 'mongoose';

const markerSchema = new mongoose.Schema({
  x: Number,
  y: Number,
  z: Number,
});

const Marker = mongoose.model('Marker', markerSchema);

export default Marker;
