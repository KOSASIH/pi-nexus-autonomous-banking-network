import express from 'express';
import { Wallet } from '../models';
import { apiUtils } from '../utils';

const router = express.Router();

router.get('/', async (req, res) => {
  try {
    const wallets = await Wallet.find().exec();
    res.json(wallets);
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.get('/:walletId', async (req, res) => {
  try {
    const wallet = await Wallet.findById(req.params.walletId).exec();
    if (!wallet) {
      res.status(404).json({ error: 'Wallet not found' });
    } else {
      res.json(wallet);
    }
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.post('/', async (req, res) => {
  try {
    const wallet = new Wallet(req.body);
    await wallet.save();
    res.json(wallet);
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.put('/:walletId', async (req, res) => {
  try {
    const wallet = await Wallet.findById(req.params.walletId).exec();
    if (!wallet) {
      res.status(404).json({ error: 'Wallet not found' });
    } else {
      wallet.set(req.body);
      await wallet.save();
      res.json(wallet);
    }
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

router.delete('/:walletId', async (req, res) => {
  try {
    await Wallet.findByIdAndRemove(req.params.walletId).exec();
    res.json({ message: 'Wallet deleted' });
  } catch (err) {
    apiUtils.errorHandler(err, req, res);
  }
});

export default router;
