const express = require('express');
const router = express.Router();
const { Device } = require('../models/Device');
const { DeviceType } = require('../models/DeviceType');
const { Firmware } = require('../models/Firmware');
const { IoTDevice } = require('../models/IoTDevice');
const { MQTTClient } = require('../utils/MQTTClient');
const { WebSocketClient } = require('../utils/WebSocketClient');
const { deviceManager } = require('../utils/deviceManager');

// Get all devices
router.get('/devices', async (req, res) => {
  try {
    const devices = await Device.find();
    res.json(devices);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting devices' });
  }
});

// Get a device by ID
router.get('/devices/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const device = await Device.findById(id);
    if (!device) {
      return res.status(404).json({ message: 'Device not found' });
    }
    res.json(device);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting device' });
  }
});

// Create a new device
router.post('/devices', async (req, res) => {
  try {
    const device = new Device(req.body);
    await device.save();
    res.json(device);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error creating device' });
  }
});

// Update a device
router.put('/devices/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const device = await Device.findById(id);
    if (!device) {
      return res.status(404).json({ message: 'Device not found' });
    }
    device.name = req.body.name;
    device.description = req.body.description;
    await device.save();
    res.json(device);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error updating device' });
  }
});

// Delete a device
router.delete('/devices/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const device = await Device.findById(id);
    if (!device) {
      return res.status(404).json({ message: 'Device not found' });
    }
    await device.remove();
    res.json({ message: 'Device deleted successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error deleting device' });
  }
});

// Get device types
router.get('/device-types', async (req, res) => {
  try {
    const deviceTypes = await DeviceType.find();
    res.json(deviceTypes);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting device types' });
  }
});

// Get firmware versions
router.get('/firmware-versions', async (req, res) => {
  try {
    const firmwareVersions = await Firmware.find();
    res.json(firmwareVersions);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting firmware versions' });
  }
});

// Update device firmware
router.put('/devices/:id/firmware', async (req, res) => {
  try {
    const id = req.params.id;
    const device = await Device.findById(id);
    if (!device) {
      return res.status(404).json({ message: 'Device not found' });
    }
    const firmwareVersion = await Firmware.findById(req.body.firmwareVersion);
    if (!firmwareVersion) {
      return res.status(404).json({ message: 'Firmware version not found' });
    }
    device.firmwareVersion = firmwareVersion;
    await device.save();
    res.json(device);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error updating device firmware' });
  }
});

// Establish MQTT connection
router.post('/mqtt-connect', async (req, res) => {
  try {
    const mqttClient = new MQTTClient();
    await mqttClient.connect();
    res.json({ message: 'MQTT connection established' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error establishing MQTT connection' });
  }
});

// Establish WebSocket connection
router.post('/ws-connect', async (req, res) => {
  try {
    const webSocketClient = new WebSocketClient();
    await webSocketClient.connect();
    res.json({ message: 'WebSocket connection established' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error establishing WebSocket connection' });
  }
});

// Get IoT device data
router.get('/iot-device-data', async (req, res) => {
  try {
    const iotDeviceData = await IoTDevice.find();
    res.json(iotDeviceData);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting IoT device data' });
  }
});

// Get device manager data
router.get('/device-manager-data', async (req, res) => {
  try {
    const deviceManagerData = await deviceManager.getDeviceManagerData();
    res.json(deviceManagerData);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error getting device manager data' });
  }
});

// Restart device
router.post('/restart-device', async (req, res) => {
  try {
    const deviceId = req.body.deviceId;
    await deviceManager.restartDevice(deviceId);
    res.json({ message: 'Device restarted successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error restarting device' });
  }
});

// Shutdown device
router.post('/shutdown-device', async (req, res) => {
  try {
    const deviceId = req.body.deviceId;
    await deviceManager.shutdownDevice(deviceId);
    res.json({ message: 'Device shut down successfully' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error shutting down device' });
  }
});

module.exports = router;
