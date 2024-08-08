import express from 'express';
import { validateRequest } from '../middleware/validation';
import { apiLimiter } from '../middleware/rateLimiter';
import { piNetworkRouter } from './piNetwork';
import { walletRouter } from './wallet';
import { authRouter } from './auth';

const router = express.Router();

router.use('/pi-network', piNetworkRouter);
router.use('/wallet', walletRouter);
router.use('/auth', authRouter);

router.get('/healthcheck', (req, res) => {
  res.json({ message: 'API is healthy' });
});

router.use(apiLimiter);
router.use(validateRequest);

export default router;
