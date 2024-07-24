import express from 'express';
import { Marker } from '../models';

const router = express.Router();

router.get('/marker', async (req, res) => {
  const marker = await Marker.findOne();
  res.json(marker);
});

export default router;
