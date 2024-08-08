import express from 'express';
import { createContract, getContracts, getContract, updateContract, deleteContract } from '../services/contractService';

const router = express.Router();

router.post('/', createContract);
router.get('/', getContracts);
router.get('/:id', getContract);
router.patch('/:id', updateContract);
router.delete('/:id', deleteContract);

export default router;
