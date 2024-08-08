// rule_based.js

class RuleBased {
  constructor() {
    this.rules = [
      {
        condition: (contract) => contract.functions.some((f) => f.name === 'transfer'),
        score: 0.5
      },
      {
        condition: (contract) => contract.functions.some((f) => f.name === 'approve'),
        score: 0.3
      },
      {
        condition: (contract) => contract.functions.some((f) => f.name === 'transferFrom'),
        score: 0.2
      }
    ];
  }

  evaluate(contract) {
    let score = 0;
    this.rules.forEach((rule) => {
      if (rule.condition(contract)) {
        score += rule.score;
      }
    });
    return score;
  }
}

module.exports = RuleBased;
