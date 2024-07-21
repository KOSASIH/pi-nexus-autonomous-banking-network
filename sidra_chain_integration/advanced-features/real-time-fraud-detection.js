const ml = require('machine-learning');

async function detectFraud(transactionInput) {
  const model = await ml.loadModel('fraud-detection-model');
  const result = await model.predict(transactionInput);
  return result;
}

module.exports = { detectFraud };
