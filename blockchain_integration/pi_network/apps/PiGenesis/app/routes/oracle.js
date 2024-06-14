import express, { Request, Response, NextFunction } from 'express';
import { getData, updateData } from '../controllers/OracleController';

const router = express.Router();

router.get('/data/:key', getData);
router.post('/data/:key', updateData);

export default router;
