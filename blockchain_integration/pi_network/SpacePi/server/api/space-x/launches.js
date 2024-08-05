const express = require('express');
const router = express.Router();
const axios = require('axios');

router.get('/', async (req, res) => {
  try {
    const response = await axios.get('https://api.spacexdata.com/v4/launches');
    const launches = response.data;
    res.json(launches);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error fetching launches' });
  }
});

router.get('/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const response = await axios.get(`https://api.spacexdata.com/v4/launches/${id}`);
    const launch = response.data;
    res.json(launch);
  } catch (error) {
    console.error(error);
    res.status(404).json({ message: 'Launch not found' });
  }
});

module.exports = router;
