import express from 'express';
import policyApi from './api/policy.api';

const router = express.Router();

router.use('/api', policyApi);

export default router;
