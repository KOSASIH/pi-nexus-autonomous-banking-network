const { AutonomousVehicle } = require('nvidia-driveworks');

class AutonomousVehicleController {
  async deployVehicle(req, res) {
    const { vehicleId, route } = req.body;
    const vehicle = new AutonomousVehicle(vehicleId);
    vehicle.setRoute(route);
    vehicle.start();
    res.json({ message: 'Vehicle deployed successfully' });
  }
}
