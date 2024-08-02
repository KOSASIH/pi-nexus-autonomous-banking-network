const plaid = require('plaid');

const PLAID_CLIENT_ID = 'YOUR_PLaid_CLIENT_ID';
const PLAID_SECRET = 'YOUR_PLaid_SECRET';
const PLAID_ENV = 'sandbox';

const plaidClient = new plaid.Client({
  clientID: PLAID_CLIENT_ID,
  secret: PLAID_SECRET,
  env: PLAID_ENV,
});

async function createOnramp(address, email, redirectUrl) {
  try {
    const onrampResponse = await plaidClient.createOnramp({
      address,
      email,
      redirectUrl,
    });
    return onrampResponse;
  } catch (error) {
    console.error(error);
    throw new Error('Failed to create on-ramp');
  }
}

async function getAccessToken() {
  try {
    const tokenResponse = await plaidClient.getToken({
      grantType: 'client_credentials',
    });
    return tokenResponse.access_token;
  } catch (error) {
    console.error(error);
    throw new Error('Failed to get access token');
  }
}

module.exports = { createOnramp, getAccessToken };
