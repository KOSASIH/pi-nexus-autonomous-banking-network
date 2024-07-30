// Eonix QC
const EonixQC = {
  // Simulator
  simulator: 'Qiskit',
  // Algorithms
  algorithms: [
    {
      name: 'Shor',
      type: 'Factorization',
      parameters: {
        n: 2048,
      },
    },
    {
      name: 'Grover',
      type: 'Search',
      parameters: {
        n: 1024,
      },
    },
  ],
};
