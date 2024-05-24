// services/CarbonFootprintCalculator.js
const carbonFootprintAPI = require('carbon-footprint-api');

const calculateCarbonFootprint = async (transaction) => {
  const apiResponse = await carbonFootprintAPI.calculateCarbonFootprint(transaction.amount, transaction.category);
  return apiResponse.carbonFootprint;
};

module.exports = { calculateCarbonFootprint };
