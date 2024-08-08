import express from 'express';
import { validateRequest } from '../middleware/validation';
import { getBalance } from '../controllers/wallet';
import { sendTransaction } from '../controllers/wallet';
import { getTransactionHistory } from '../controllers/wallet';

const router = express.Router();

router.get('/balance', validateRequest, getBalance);
router.post('/send-transaction', validateRequest, sendTransaction);
router.get('/transaction-history', validateRequest, getTransactionHistory);

export default router;
