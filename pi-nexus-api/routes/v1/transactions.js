import express from 'express';
import { Transaction } from '../models';
import { apiUtils } from '../utils';

const router = express.Router();

router.get('/', async (req, res) => {
  try {
    const transactions = await Transaction.find().exec();
    res.json(transactions);
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.get('/:transactionId', async (req, res) => {
  try {
    const transaction = await Transaction.findById(req.params.transactionId).exec();
    if (!transaction) {
      res.status(404).json({ error: 'Transaction not found' });
    } else {
      res.json(transaction);
    }
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.post('/', async (req, res) => {
  try {
    const transaction = new Transaction(req.body);
    await transaction.save();
    res.json(transaction);
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

export default router;
