import express from 'express';
import { getTransactions, getTransactionDetails } from '../services/TransactionService';
import { authenticate } from '../middleware/auth';

const router = express.Router();

router.get('/transactions', authenticate, getTransactions);
router.get('/transactions/:id', authenticate, getTransactionDetails);

router.post('/transactions', authenticate, (req, res) => {
  const { amount, description } = req.body;
  const transaction = { amount, description, userId: req.user.id };
  TransactionService.createTransaction(transaction)
    .then((transaction) => res.json(transaction))
    .catch((error) => res.status(500).json({ message: error.message }));
});

router.put('/transactions/:id', authenticate, (req, res) => {
  const { id } = req.params;
  const { amount, description } = req.body;
  TransactionService.updateTransaction(id, { amount, description })
    .then((transaction) => res.json(transaction))
    .catch((error) => res.status(500).json({ message: error.message }));
});

router.delete('/transactions/:id', authenticate, (req, res) => {
  const { id } = req.params;
  TransactionService.deleteTransaction(id)
    .then(() => res.json({ message: 'Transaction deleted successfully' }))
    .catch((error) => res.status(500).json({ message: error.message }));
});

export default router;
