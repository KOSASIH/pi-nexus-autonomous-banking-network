import express from 'express';
import { checkSystemHealth } from '../utils/systemHealth';

const router = express.Router();

router.get('/health', (req, res) => {
    const healthStatus = checkSystemHealth();
    res.status(200).json(healthStatus);
});

export default router;
