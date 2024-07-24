import express from 'express';
import { trainModel, predict } from '../utils/ai-utils';

const router = express.Router();

router.post('/train-model', async (req, res) => {
  try {
    const model = await trainModel();
    res.json({ message: 'Model trained successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

router.post('/predict', async (req, res) => {
  try {
    const inputData = req.body.inputData;
    const output = await predict(inputData);
    res.json({ prediction: output });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

export default router;
