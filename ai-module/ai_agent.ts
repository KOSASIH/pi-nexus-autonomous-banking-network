import * as tf from '@tensorflow/tfjs';
import { Brain } from 'brain.js';

interface Transaction {
  id: string;
  timestamp: number;
  amount: number;
  sender: string;
  recipient: string;
}

class AIAgent {
  private brain: Brain;
  private transactionData: Transaction[];

  constructor() {
    this.brain = new Brain();
    this.transactionData = [];
  }

  async train(data: Transaction[]) {
    this.transactionData = data;
    const input = this.preprocessData(data);
    const output = this.createOutput(data);
    this.brain.train(input, output);
  }

  async analyzeTransaction(transaction: Transaction) {
    const input = this.preprocessTransaction(transaction);
    const output = this.brain.run(input);
    return this.postprocessOutput(output);
  }

  private preprocessData(data: Transaction[]) {
    const inputs = [];
    for (const transaction of data) {
      const input = [
        transaction.timestamp,
        transaction.amount,
        transaction.sender.length,
        transaction.recipient.length,
      ];
      inputs.push(input);
    }
    return inputs;
  }

  private createOutput(data: Transaction[]) {
    const outputs = [];
    for (const transaction of data) {
      const output = [transaction.amount > 100 ? 1 : 0]; // anomaly detection
      outputs.push(output);
    }
    return outputs;
  }

  private preprocessTransaction(transaction: Transaction) {
    const input = [
      transaction.timestamp,
      transaction.amount,
      transaction.sender.length,
      transaction.recipient.length,
    ];
    return input;
  }

  private postprocessOutput(output: number[]) {
    return output[0] > 0.5 ? true : false; // anomaly detected
  }
}

export default AIAgent;
