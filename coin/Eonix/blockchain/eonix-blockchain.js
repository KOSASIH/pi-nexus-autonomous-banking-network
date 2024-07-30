// Eonix Blockchain
const EonixBlockchain = {
  // Network
  network: {
    name: 'EonixNetwork',
    protocol: 'TCP/IP',
    port: 8080,
  },
  // Consensus
  consensus: {
    algorithm: 'ProofOfStake',
    validators: ['Validator1', 'Validator2'],
  },
  // Block
  block: {
    structure: {
      header: {
        version: 1,
        timestamp: Date.now(),
        prevBlockHash: '0x0000000000000000000000000000000000000000000000000000000000000000',
        merkleRoot: '0x0000000000000000000000000000000000000000000000000000000000000000',
      },
      transactions: [],
    },
  },
};
