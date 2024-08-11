export class NexusError extends Error {
  constructor(message) {
    super(message);
    this.name = 'NexusError';
  }
}

export class BlockchainError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BlockchainError';
  }
}
