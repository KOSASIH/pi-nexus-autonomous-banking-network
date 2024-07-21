const ai = require('ai-library');

async function processTransaction(transactionInput) {
  const model = await ai.loadModel('transaction-processing-model');
  const result = await model.predict(transactionInput);
  return result;
}

module.exports = { processTransaction };
