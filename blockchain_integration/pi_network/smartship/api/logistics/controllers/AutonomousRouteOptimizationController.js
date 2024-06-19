const orTools = require('ortools');

class AutonomousRouteOptimizationController {
  async optimizeRoute(req, res) {
    const { vehicles, packages, depots, timeWindows } = req.body;
    const data = new orTools.RoutingModel();
    data.AddDimension('distance', 0, 10000, true, null, null);
    data.AddDimension('time', 0, 10000, true, null, null);
    //...
    const solution = orTools.routing.Solve(data);
    res.json({ solution });
  }
}
