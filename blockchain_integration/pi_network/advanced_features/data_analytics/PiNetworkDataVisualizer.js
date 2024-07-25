// Importing necessary libraries
import * as d3 from 'd3-array';
import * as Plot from '@observablehq/plot';
import { PiNetworkData } from './PiNetworkData';

// Class for data visualization
class PiNetworkDataVisualizer {
  constructor(data) {
    this.data = data;
  }

  // Function to visualize user transaction data
  visualizeTransactions() {
    const transactions = this.data.getUserTransactions();
    const chart = Plot.plot({
      marks: [
        Plot.line(transactions, { x: 'date', y: 'amount' }),
        Plot.point(transactions, { x: 'date', y: 'amount' }),
      ],
      width: 800,
      height: 400,
    });
    document.getElementById('transactions-chart').appendChild(chart);
  }

  // Function to visualize user behavior data
  visualizeBehavior() {
    const behavior = this.data.getUserBehavior();
    const chart = Plot.plot({
      marks: [
        Plot.bar(behavior, { x: 'category', y: 'frequency' }),
      ],
      width: 800,
      height: 400,
    });
    document.getElementById('behavior-chart').appendChild(chart);
  }

  // Function to visualize market trends data
  visualizeMarketTrends() {
    const trends = this.data.getMarketTrends();
    const chart = Plot.plot({
      marks: [
        Plot.line(trends, { x: 'date', y: 'value' }),
      ],
      width: 800,
      height: 400,
    });
    document.getElementById('market-trends-chart').appendChild(chart);
  }
}

// Exporting the class
export { PiNetworkDataVisualizer };
