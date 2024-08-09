// environment.ts

import { join } from 'path';
import { readFileSync } from 'fs';
import { Config } from './config.interface';
import { QuantumCryptoConfig } from './quantum-crypto/quantum-crypto.config';
import { AIMLConfig } from './ai-ml/ai-ml.config';
import { NodeConfig } from './node/node.config';
import { BlockchainConfig } from './blockchain/blockchain.config';
import { QuantumComputingConfig } from './quantum-computing/quantum-computing.config';

const envFile = join(__dirname, '..', '..', '.env');
const envConfig = readFileSync(envFile, 'utf8').split('\n').reduce((acc, line) => {
  const [key, value] = line.split('=');
  acc[key.trim()] = value.trim();
  return acc;
}, {});

const config: Config = {
  environment: envConfig.NODE_ENV || 'development',
  quantumCrypto: new QuantumCryptoConfig({
    keySize: parseInt(envConfig.QUANTUM_CRYPTO_KEY_SIZE, 10) || 2048,
    signatureScheme: envConfig.QUANTUM_CRYPTO_SIGNATURE_SCHEME || 'ECDSA',
    hashFunction: envConfig.QUANTUM_CRYPTO_HASH_FUNCTION || 'SHA-256',
  }),
  aiML: new AIMLConfig({
    modelPath: envConfig.AI_ML_MODEL_PATH || 'models/',
    dataPath: envConfig.AI_ML_DATA_PATH || 'data/',
    batchSize: parseInt(envConfig.AI_ML_BATCH_SIZE, 10) || 32,
    epochs: parseInt(envConfig.AI_ML_EPOCHS, 10) || 10,
  }),
  node: new NodeConfig({
    port: parseInt(envConfig.NODE_PORT, 10) || 3000,
    host: envConfig.NODE_HOST || 'localhost',
    protocol: envConfig.NODE_PROTOCOL || 'http',
  }),
  blockchain: new BlockchainConfig({
    network: envConfig.BLOCKCHAIN_NETWORK || 'mainnet',
    nodeUrl: envConfig.BLOCKCHAIN_NODE_URL || 'https://node.pi-network.io',
    contractAddress: envConfig.BLOCKCHAIN_CONTRACT_ADDRESS || '0x...',
  }),
  quantumComputing: new QuantumComputingConfig({
    provider: envConfig.QUANTUM_COMPUTING_PROVIDER || 'IBM Quantum',
    apiKey: envConfig.QUANTUM_COMPUTING_API_KEY || '...',
    backend: envConfig.QUANTUM_COMPUTING_BACKEND || 'ibmq_qasm_simulator',
  }),
};

export default config;
