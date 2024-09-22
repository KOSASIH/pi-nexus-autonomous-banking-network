import * as tf from '@tensorflow/tfjs';

interface Transaction {
  id: string;
  timestamp: number;
  amount: number;
  sender: string;
  recipient: string;
}

class AIUtils {
  static async loadTransactions(filename: string): Promise<Transaction[]> {
    const data = await fetch(filename);
    const json = await data.json();
    return json.transactions;
  }

  static async preprocessTransactions(transactions: Transaction[]) {
    const inputs = [];
    for (const transaction of transactions) {
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

  static async featureEngineering(transactions: Transaction[]) {
    const features = [];
    for (const transaction of transactions) {
      const feature = [
        transaction.amount,
        transaction.sender.length,
        transaction.recipient.length,
      ];
      features.push(feature);
    }
    return features;
  }

  static async evaluateModel(model: tf.Model, data: Transaction[]) {
    const inputs = AIUtils.preprocessTransactions(data);
    const outputs = model.predict(inputs);
    const metrics = tf.metrics.meanSquaredError(outputs, inputs);
    return metrics;
  }
}

export default AIUtils;
