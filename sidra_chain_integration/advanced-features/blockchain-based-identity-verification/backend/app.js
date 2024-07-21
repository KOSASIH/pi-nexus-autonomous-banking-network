const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const sidraChain = require('./sidraChain');

app.use(bodyParser.json());

app.post('/api/verify', async (req, res) => {
  const { userAddress, userData } = req.body;
  try {
    // Call the Sidra Chain API to verify user data
    const response = await sidraChain.verifyIdentity(userAddress, userData);
    if (response.verified) {
      res.json({ verified: true });
    } else {
      res.json({ verified: false });
    }
  } catch (error) {
    console.error(`Error verifying user: ${error}`);
    res.status(500).json({ error: 'Failed to verify user' });
  }
});

app.listen(3001, () => {
  console.log('Server listening on port 3001');
});
