const ml = require('machine-learning');

async function assessRisk(transactionInput) {
  const model = await ml.loadModel('risk-assessment-model');
  const result = await model.predict(transactionInput);
  return result;
}

module.exports = { assessRisk };
