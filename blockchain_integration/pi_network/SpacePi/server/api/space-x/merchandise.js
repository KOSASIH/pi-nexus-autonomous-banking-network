const express = require('express');
const router = express.Router();
const merchandiseData = require('../data/merchandise');

router.get('/', (req, res) => {
  res.json(merchandiseData);
});

router.get('/:id', (req, res) => {
  const id = req.params.id;
  const merchandise = merchandiseData.find(item => item.id === id);
  if (merchandise) {
    res.json(merchandise);
  } else {
    res.status(404).json({ message: 'Merchandise not found' });
  }
});

module.exports = router;
