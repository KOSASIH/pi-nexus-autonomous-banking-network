import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

class RealtimeChart {
  constructor(element) {
    this.chartElement = element;
    this.data = [];
    this.chart = (
      <LineChart width={600} height={300} data={this.data}>
        <Line type="monotone" dataKey="value" stroke="#8884d8" />
        <XAxis dataKey="time" />
        <YAxis />
        <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
        <Tooltip />
      </LineChart>
    );
  }

  updateData(newData) {
    this.data = [...this.data, ...newData];
    this.chartElement.innerHTML = '';
    this.chartElement.appendChild(this.chart);
  }
}

export default RealtimeChart;
