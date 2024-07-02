import React from 'react';
import { Chart } from 'chart.js';

const ChartComponent = ({ data }) => {
  return (
    <div>
      <Chart type="line" data={data} />
    </div>
  );
};

export default ChartComponent;
