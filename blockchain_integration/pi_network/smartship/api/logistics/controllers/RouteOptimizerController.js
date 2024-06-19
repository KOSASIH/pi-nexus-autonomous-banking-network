const orTools = require('ortools');

class RouteOptimizerController {
  async optimizeRoute(req, res) {
    const { origin, destination, packages } = req.body;
    const data = new orTools RoutingModel();
    data.AddDimension('distance', 0, 3000, true, null, null);
    data.AddDimension('time', 0, 8 * 60 * 60, true, null, null);
    // ...
    const solution = orTools routing.Solve(data);
    res.json(solution);
  }
}
