import express from 'express';
import piNetworkApi from '../api/piNetwork';

const router = express.Router();

router.get('/balance', async (req, res) => {
  const balance = await piNetworkApi.getPiBalance(req.user.address);
  res.json({ balance });
});

router.post('/transaction', async (req, res) => {
  const { amount, recipient } = req.body;
  const transaction = await piNetworkApi.sendPiTransaction(req.user.address, recipient, amount);
  res.json({ transaction });
});

export default router;
