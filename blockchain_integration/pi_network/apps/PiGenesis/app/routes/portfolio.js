import express, { Request, Response, NextFunction } from 'express';
import { authorizeMiddleware } from '../middleware/authMiddleware';
import { getPortfolio, deposit, withdraw } from '../controllers/PortfolioController';

const router = express.Router();

router.get('/portfolio', authorizeMiddleware, getPortfolio);
router.post('/deposit', authorizeMiddleware, deposit);
router.post('/withdraw', authorizeMiddleware, withdraw);

export default router;
