import express from 'express';
import { createPolicy, getPolicies, getPolicy, updatePolicy, deletePolicy } from '../controllers/policy.controller';

const router = express.Router();

router.post('/policies', createPolicy);
router.get('/policies', getPolicies);
router.get('/policies/:id', getPolicy);
router.put('/policies/:id', updatePolicy);
router.delete('/policies/:id', deletePolicy);

export default router;
