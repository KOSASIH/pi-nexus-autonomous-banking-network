const express = require('express');
const router = express.Router();
const sidraChain = require('../utils/sidraChain');

router.post('/verify', async (req, res) => {
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

module.exports = router;
