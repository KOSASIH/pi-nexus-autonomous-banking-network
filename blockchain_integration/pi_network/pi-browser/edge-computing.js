import { EdgeCompute } from 'edge-compute-sdk';

class EdgeComputing {
  constructor() {
    this.edgeCompute = new EdgeCompute();
  }

  async processDataAtEdge(data) {
    const processedData = await this.edgeCompute.process(data);
    return processedData;
  }
}

export default EdgeComputing;
