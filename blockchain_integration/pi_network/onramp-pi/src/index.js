const express = require('express');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const axios = require('axios');
const plaid = require('plaid');

const app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

mongoose.connect('mongodb://localhost/onramp-pi', { useNewUrlParser: true, useUnifiedTopology: true });

const fundWallet = async () => {
  // Get user's wallet address and email address
  const walletAddress = user?.wallet?.address;
  const emailAddress = user?.email?.address;

  // Get current URL
  const currentUrl = window.location.href;

  // Get Privy auth token
  const authToken = await getAccessToken();

  // Send request to server with these details
  try {
    const onrampResponse = await axios.post(
      '/api/onramp',
      {
        address: walletAddress,
        email: emailAddress,
        redirectUrl: currentUrl,
      },
      {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
      }
    );
    return onrampResponse.data.url as string;
  } catch (error) {
    console.error(error);
    return undefined;
  }
};

app.post('/api/onramp', async (req, res) => {
  // Handle on-ramp flow
  const { address, email, redirectUrl } = req.body;
  // Implement on-ramp logic using Plaid
  const plaidClient = new plaid.Client({
    clientID: 'YOUR_PLaid_CLIENT_ID',
    secret: 'YOUR_PLaid_SECRET',
    env: 'sandbox',
  });
  const onrampResponse = await plaidClient.createOnramp({
    address,
    email,
    redirectUrl,
  });
  res.json(onrampResponse);
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
