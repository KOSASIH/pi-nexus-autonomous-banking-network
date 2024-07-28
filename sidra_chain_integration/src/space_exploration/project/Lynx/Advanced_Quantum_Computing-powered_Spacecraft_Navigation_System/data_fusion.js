import { SensorData } from 'sensor-data-library';
import { GPSData } from 'gps-data-library';
import { AstronomicalData } from 'astronomical-data-library';

class DataFusion {
  constructor() {
    this.sensorData = new SensorData();
    this.gpsData = new GPSData();
    this.astronomicalData = new AstronomicalData();
  }

  fuse(data) {
    // Fuse data from various sources into a single dataset
    const fusedData = this.sensorData.fuse(data.sensorData);
    fusedData = this.gpsData.fuse(fusedData);
    fusedData = this.astronomicalData.fuse(fusedData);
    return fusedData;
  }
}

export default DataFusion;
