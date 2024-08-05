// Eonix AI
const EonixAI = {
  // Engine
  engine: 'TensorFlow',
  // Models
  models: [
    {
      name: 'LSTM',
      architecture: 'LSTM',
      layers: [
        {
          type: 'LSTM',
          units: 128,
        },
      ],
    },
    {
      name: 'GRU',
      architecture: 'GRU',
      layers: [
        {
          type: 'GRU',
          units: 128,
        },
      ],
    },
  ],
};
