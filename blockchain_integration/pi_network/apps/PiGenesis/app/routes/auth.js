import express, { Request, Response, NextFunction } from 'express';
import { authenticateMiddleware, authorizeMiddleware } from '../middleware/authMiddleware';
import { login, register, logout } from '../controllers/AuthController';

const router = express.Router();

router.post('/login', authenticateMiddleware, login);
router.post('/register', register);
router.post('/logout', authorizeMiddleware, logout);

export default router;
