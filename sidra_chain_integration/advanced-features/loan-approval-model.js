// loan-approval-model.js
const tf = require('@tensorflow/tfjs');

class LoanApprovalModel {
  constructor() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 10, inputShape: [3] }));
    this.model.add(tf.layers.dense({ units: 1 }));
    this.model.compile({ optimizer: tf.optimizers.adam(), loss: 'meanSquaredError' });
  }

  async train(data) {
    const xs = data.map((d) => [d.creditScore, d.loanAmount, d.interestRate]);
    const ys = data.map((d) => d.approvalStatus);
    this.model.fit(xs, ys, { epochs: 100 });
  }

  async predict(loanApplication) {
    const input = [loanApplication.creditScore, loanApplication.loanAmount, loanApplication.interestRate];
    const output = this.model.predict(input);
    return output.dataSync()[0] > 0.5 ? 'approved' : 'rejected';
  }
}

module.exports = LoanApprovalModel;
