import * as React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

interface DashboardProps {
  data: any[];
}

class AnalyticsDashboard extends React.Component<DashboardProps, {}> {
  render() {
    return (
      <div>
        <h1>Network Performance Dashboard</h1>
        <LineChart width={800} height={400} data={this.props.data}>
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
          <XAxis dataKey="timestamp" />
          <YAxis />
          <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
          <Tooltip />
        </LineChart>
      </div>
    );
  }
}

export default AnalyticsDashboard;
