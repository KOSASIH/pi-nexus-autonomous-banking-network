import express from 'express';
import { Contract } from '../models';
import { apiUtils } from '../utils';

const router = express.Router();

router.get('/', async (req, res) => {
  try {
    const contracts = await Contract.find().exec();
    res.json(contracts);
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.get('/:contractId', async (req, res) => {
  try {
    const contract = await Contract.findById(req.params.contractId).exec();
    if (!contract) {
      res.status(404).json({ error: 'Contract not found' });
    } else {
      res.json(contract);
    }
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.post('/', async (req, res) => {
  try {
    const contract = new Contract(req.body);
    await contract.save();
    res.json(contract);
  }
router.post('/', async (req, res) => {
  try {
    const contract = new Contract(req.body);
    await contract.save();
    res.json(contract);
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.put('/:contractId', async (req, res) => {
  try {
    const contract = await Contract.findById(req.params.contractId).exec();
    if (!contract) {
      res.status(404).json({ error: 'Contract not found' });
    } else {
      contract.set(req.body);
      await contract.save();
      res.json(contract);
    }
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.delete('/:contractId', async (req, res) => {
  try {
    await Contract.findByIdAndRemove(req.params.contractId).exec();
    res.json({ message: 'Contract deleted' });
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

export default router;
