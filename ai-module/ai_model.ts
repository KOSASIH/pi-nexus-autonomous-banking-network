import * as tf from '@tensorflow/tfjs';

interface Transaction {
  id: string;
  timestamp: number;
  amount: number;
  sender: string;
  recipient: string;
}

class AIModel {
  private model: tf.Sequential;

  constructor() {
    this.model = this.createModel();
  }

  async train(data: Transaction[]) {
    const inputs = this.preprocessData(data);
    const outputs = this.createOutput(data);
    this.model.fit(inputs, outputs, { epochs: 10 });
  }

  async predict(transaction: Transaction) {
    const input = this.preprocessTransaction(transaction);
    const output = this.model.predict(input);
    return this.postprocessOutput(output);
  }

  private createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, inputShape: [4] }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.optimizers.adam(), loss: 'meanSquaredError' });
    return model;
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
      const output = [transaction.amount];
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
    return output[0];
  }
}

export default AIModel;
