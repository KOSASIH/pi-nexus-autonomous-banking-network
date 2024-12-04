// config/apiKeys.js

const environment = require('./environment');

const apiKeys = {
    STRIPE_API_KEY: process.env.STRIPE_API_KEY || 'your_stripe_api_key',
    TWILIO_ACCOUNT_SID: process.env.TWILIO_ACCOUNT_SID || 'your_twilio_account_sid',
    TWILIO_AUTH_TOKEN: process.env.TWILIO_AUTH_TOKEN || 'your_twilio_auth_token',
    BINANCE_API_KEY: process.env.BINANCE_API_KEY || 'your_binance_api_key',
};

module.exports = apiKeys;
