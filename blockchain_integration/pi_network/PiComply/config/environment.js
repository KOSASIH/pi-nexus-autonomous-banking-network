const dotenv = require('dotenv');
const fs = require('fs');
const path = require('path');

dotenv.config();

const config = require('./config.json');

const environment = {
  NETWORK: process.env.NETWORK || config.network.mainnet,
  CONTRACTS: config.contracts,
  ORACLES: config.oracles,
  AI_COMPLIANCE_ENGINE: config.aiComplianceEngine,
  REPORTING_ANALYTICS: config.reportingAnalytics,
  SECURITY: config.security
};

module.exports = environment;
