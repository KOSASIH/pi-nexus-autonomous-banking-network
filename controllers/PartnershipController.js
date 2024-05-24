// controllers/PartnershipController.js
const PartnershipService = require('../services/PartnershipService');

const getPartnerships = async (req, res) => {
  const partnerships = await PartnershipService.getPartnerships();
  res.send(partnerships);
};

module.exports = { getPartnerships };
