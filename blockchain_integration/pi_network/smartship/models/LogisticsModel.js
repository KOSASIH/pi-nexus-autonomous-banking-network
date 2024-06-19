const mongoose = require('mongoose');
const { Schema } = mongoose;
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { v4: uuidv4 } = require('uuid');
const { LogisticsContract } = require('../blockchain_integration/contracts');

const logisticsContract = new LogisticsContract();

const shipmentSchema = new Schema({
  _id: { type: String, required: true, unique: true, default: uuidv4 },
  sender: { type: String, required: true },
  recipient: { type: String, required: true },
  shipmentType: { type: String, required: true, enum: ['PACKAGE', 'DOCUMENT', 'FREIGHT'] },
  weight: { type: Number, required: true },
  dimensions: { type: [Number], required: true },
  trackingNumber: { type: String, required: true, unique: true },
  status: { type: String, required: true, enum: ['PENDING', 'IN_TRANSIT', 'DELIVERED'] },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

shipmentSchema.pre('save', async function(next) {
  try {
    const shipment = this;
    const shipmentId = shipment._id;
    await logisticsContract.createShipment(shipment);
    next();
  } catch (error) {
    next(error);
  }
});

shipmentSchema.methods.generateToken = function() {
  const shipment = this;
  const token = jwt.sign({ shipmentId: shipment._id }, process.env.SECRET_KEY, {
    expiresIn: '1h'
  });
  return token;
};

shipmentSchema.methods.updateStatus = async function(status) {
  try {
    const shipment = this;
    await logisticsContract.updateShipmentStatus(shipment._id, status);
    shipment.status = status;
    await shipment.save();
  } catch (error) {
    throw error;
  }
};

shipmentSchema.methods.addTrackingNumber = async function(trackingNumber) {
  try {
    const shipment = this;
    await logisticsContract.addShipmentTracking(shipment._id, trackingNumber);
    shipment.trackingNumber = trackingNumber;
    await shipment.save();
  } catch (error) {
    throw error;
  }
};

const LogisticsModel = mongoose.model('Logistics', shipmentSchema);

module.exports = LogisticsModel;
