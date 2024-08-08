// node.config.js

module.exports = {
  // Node configuration
  node: {
    // Node ID
    id: 'node-1',
    // Node type (e.g. validator, observer, etc.)
    type: 'validator',
    // Node network configuration
    network: {
      // Node network ID
      id: 'network-1',
      // Node network protocol (e.g. TCP, UDP, etc.)
      protocol: 'tcp',
      // Node network address
      address: '0.0.0.0',
      // Node network port
      port: 8080
    },
    // Node consensus configuration
    consensus: {
      // Consensus algorithm (e.g. PBFT, Raft, etc.)
      algorithm: 'pbft',
      // Consensus timeout
      timeout: 10000,
      // Consensus max retries
      maxRetries: 5
    },
    // Node storage configuration
    storage: {
      // Storage type (e.g. memory, disk, etc.)
      type: 'memory',
      // Storage capacity
      capacity: 1000000
    },
    // Node security configuration
    security: {
      // Node private key
      privateKey: 'path/to/private/key',
      // Node public key
      publicKey: 'path/to/public/key',
      // Node encryption algorithm (e.g. AES, RSA, etc.)
      encryptionAlgorithm: 'aes-256-gcm'
    }
  },

  // Blockchain configuration
  blockchain: {
    // Blockchain ID
    id: 'blockchain-1',
    // Blockchain network ID
    networkId: 'network-1',
    // Blockchain genesis block
    genesisBlock: {
      // Genesis block timestamp
      timestamp: 1643723400,
      // Genesis block transactions
      transactions: []
    },
    // Blockchain block time
    blockTime: 10000,
    // Blockchain block size
    blockSize: 1000000
  },

  // Logger configuration
  logger: {
    // Logger level (e.g. debug, info, warn, error)
    level: 'info',
    // Logger output (e.g. console, file, etc.)
    output: 'console'
  }
};
