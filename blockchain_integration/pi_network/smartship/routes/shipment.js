const express = require('express');
const router = express.Router();
const { validate } = require('jsonschema');
const { shipmentSchema } = require('../models/schemas');
const { LogisticsModel } = require('../models/LogisticsModel');
const { authenticateToken } = require('../middleware/auth');

router.use(authenticateToken);

router.post('/createShipment', async (req, res) => {
  const { sender, recipient, shipmentType, weight, dimensions, trackingNumber } = req.body;

  try {
    await validate(req.body, shipmentSchema);
  } catch (error) {
    return res.status(400).json({ error: 'Invalid request body' });
  }

  try {
    const shipment = new LogisticsModel({
      sender,
      recipient,
      shipmentType,
      weight,
      dimensions,
      trackingNumber
    });
    await shipment.save();
    res.json({ message: 'Shipment created successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create shipment' });
  }
});

router.get('/getShipment/:shipmentId', async (req, res) => {
  const { shipmentId } = req.params;

  try {
    const shipment = await LogisticsModel.findById(shipmentId);
    if (!shipment) {
      return res.status(404).json({ error: 'Shipment not found' });
    }
    res.json(shipment);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to retrieve shipment' });
  }
});

router.get('/getShipments', async (req, res) => {
  try {
    const shipments = await LogisticsModel.find();
    res.json(shipments);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to retrieve shipments' });
  }
});

router.put('/updateShipment/:shipmentId', async (req, res) => {
  const { shipmentId } = req.params;
  const { status } = req.body;

  try {
    const shipment = await LogisticsModel.findById(shipmentId);
    if (!shipment) {
      return res.status(404).json({ error: 'Shipment not found' });
    }
    shipment.status = status;
    await shipment.save();
    res.json({ message: 'Shipment updated successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to update shipment' });
  }
});

router.delete('/deleteShipment/:shipmentId', async (req, res) => {
  const { shipmentId } = req.params;

  try {
    await LogisticsModel.findByIdAndRemove(shipmentId);
    res.json({ message: 'Shipment deleted successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to delete shipment' });
  }
});

module.exports = router;
