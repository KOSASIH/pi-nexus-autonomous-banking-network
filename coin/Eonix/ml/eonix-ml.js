// Eonix ML
const EonixML = {
  // Algorithms
  algorithms: [
    {
      name: 'LinearRegression',
      type: 'Regression',
            parameters: {
        learningRate: 0.01,
        regularization: 'L2',
      },
    },
    {
      name: 'DecisionTree',
      type: 'Classification',
      parameters: {
        maxDepth: 10,
        minSamplesSplit: 2,
      },
    },
  ],
  // Datasets
  datasets: [
    {
      name: 'EonixMarketData',
      type: 'TimeSeries',
      features: ['open', 'high', 'low', 'close'],
      target: 'close',
    },
  ],
};
