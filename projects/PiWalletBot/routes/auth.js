import express from 'express';
import { validateRequest } from '../middleware/validation';
import { authenticate } from '../controllers/auth';
import { register } from '../controllers/auth';
import { forgotPassword } from '../controllers/auth';
import { resetPassword } from '../controllers/auth';

const router = express.Router();

router.post('/register', validateRequest, register);
router.post('/login', validateRequest, authenticate);
router.post('/forgot-password', validateRequest, forgotPassword);
router.post('/reset-password', validateRequest, resetPassword);

export default router;
