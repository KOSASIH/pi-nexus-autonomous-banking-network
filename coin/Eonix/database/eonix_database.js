// Eonix Database
const EonixDatabase = {
  // Type
  type: 'GraphDB',
  // Storage
  storage: 'IPFS',
  // Schema
  schema: {
    nodes: [
      {
        id: 'EonixNode',
        properties: {
          name: 'Eonix',
          type: 'Node',
        },
      },
    ],
    edges: [
      {
        id: 'EonixEdge',
        properties: {
          name: 'EonixEdge',
          type: 'Edge',
        },
      },
    ],
  },
};
