const autonomousVehicleUtils = {
  calculateRoute: (start, end) => {
    const route = [];
    const graph = new Graph();
    const startNode = graph.addNode(start.latitude, start.longitude);
    const endNode = graph.addNode(end.latitude, end.longitude);
    const shortestPath = graph.shortestPath(startNode, endNode);
    shortestPath.forEach((node) => {
      route.push({ latitude: node.latitude, longitude: node.longitude });
    });
    return route;
  },

  updateVehicleState: (vehicle, sensorData) => {
    const { speed, acceleration, steeringAngle } = sensorData;
    vehicle.speed = speed;
    vehicle.acceleration = acceleration;
    vehicle.steeringAngle = steeringAngle;
    vehicle.updateState();
  },

  calculateDistance: (lat1, lon1, lat2, lon2) => {
    const R = 6371; // radius of the earth in km
    const dLat = deg2rad(lat2 - lat1);
    const dLon = deg2rad(lon2 - lon1);
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const distance = R * c;
    return distance;
  },

  deg2rad: (deg) => {
    return deg * (Math.PI / 180);
  },
};

class Graph {
  constructor() {
    this.nodes = {};
  }

  addNode(lat, lon) {
    const node = new Node(lat, lon);
    this.nodes[`${lat},${lon}`] = node;
    return node;
  }

  shortestPath(startNode, endNode) {
    const queue = [startNode];
    const distances = {};
    const previousNodes = {};
    distances[startNode] = 0;
    while (queue.length > 0) {
      const currentNode = queue.shift();
      if (currentNode === endNode) {
        break;
      }
      const neighbors = this.getNeighbors(currentNode);
      neighbors.forEach((neighbor) => {
        const distance = distances[currentNode] + this.calculateDistance(currentNode.latitude, currentNode.longitude, neighbor.latitude, neighbor.longitude);
        if (!distances[neighbor] || distance < distances[neighbor]) {
          distances[neighbor] = distance;
          previousNodes[neighbor] = currentNode;
          queue.push(neighbor);
        }
      });
    }
    const shortestPath = [];
    let currentNode = endNode;
    while (currentNode !== startNode) {
      shortestPath.unshift(currentNode);
      currentNode = previousNodes[currentNode];
    }
    shortestPath.unshift(startNode);
    return shortestPath;
  }

  getNeighbors(node) {
    const neighbors = [];
    for (const key in this.nodes) {
      const neighbor = this.nodes[key];
      if (neighbor !== node && this.calculateDistance(node.latitude, node.longitude, neighbor.latitude, neighbor.longitude) < 1) {
        neighbors.push(neighbor);
      }
    }
    return neighbors;
  }
}

class Node {
  constructor(lat, lon) {
    this.latitude = lat;
    this.longitude = lon;
  }
}

export default autonomousVehicleUtils;
