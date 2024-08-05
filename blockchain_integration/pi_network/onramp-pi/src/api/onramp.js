const express = require('express');
const router = express.Router();
const plaid = require('plaid');

router.post('/', async (req, res) => {
  const { address, email, redirectUrl } = req.body;
  const plaidClient = new plaid.Client({
    clientID: 'YOUR_PLaid_CLIENT_ID',
    secret: 'YOUR_PLaid_SECRET',
    env: 'sandbox',
  });
  try {
    const onrampResponse = await plaidClient.createOnramp({
      address,
      email,
      redirectUrl,
    });
    res.json(onrampResponse);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to create on-ramp' });
  }
});

module.exports = router;
