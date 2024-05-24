import { MachineLearningModel } from 'ml-model';
import { Transaction } from './Transaction';
import { FinancialGoal } from './FinancialGoal';

class FinancialAdvisor {
  constructor() {
    this.model = new MachineLearningModel('financial_advisor_model');
  }

  async train() {
    const transactions = await Transaction.find().exec();
    const financialGoals = await FinancialGoal.find().exec();
    const trainingData = transactions.map((transaction) => ({
      input: transaction.amount,
      output: financialGoals.find((goal) => goal.category === transaction.category).targetAmount
    }));
    this.model.train(trainingData);
  }

  async predict(transaction) {
    const input = [transaction.amount];
    const output = this.model.predict(input);
    return output;
  }
}

export default FinancialAdvisor;
