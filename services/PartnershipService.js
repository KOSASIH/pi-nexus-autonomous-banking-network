// services/PartnershipService.js
const partnershipAPI = require('partnership-api');

const getPartnerships = async () => {
  const apiResponse = await partnershipAPI.getPartnerships();
  return apiResponse.partnerships;
};

module.exports = { getPartnerships };
