const { IoTDevice } = require('aws-iot-device-sdk');

class SupplyChainVisibilityController {
  async trackShipment(req, res) {
    const { shipmentId } = req.body;
    const device = new IoTDevice(shipmentId);
    const locationData = device.getLocation();
    res.json({ locationData });
  }
}
