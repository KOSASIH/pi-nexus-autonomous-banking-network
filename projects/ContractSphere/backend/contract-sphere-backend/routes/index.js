import express from 'express';
import contractRoutes from './contractRoutes';
import userRoutes from './userRoutes';

const router = express.Router();

router.use('/contracts', contractRoutes);
router.use('/users', userRoutes);

export default router;
