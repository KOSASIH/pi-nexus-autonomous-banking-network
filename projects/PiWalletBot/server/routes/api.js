import express from 'express';
import walletApi from '../api/wallet';

const router = express.Router();

router.post('/wallet', walletApi);

export default router;
