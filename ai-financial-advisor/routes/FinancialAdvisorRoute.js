const express = require('express');
const router = express.Router();
const NLPController = require('../controllers/NLPController');

router.post('/getAdvice', async (req, res) => {
  const text = req.body.text;
  const userId = req.user.id;
  const advice = await NLPController.getAdvice(text, userId);
  res.json({ advice });
});

router.post('/setGoal', async (req, res) => {
  const text = req.body.text;
  const userId = req.user.id;
  const goal = await NLPController.setGoal(text, userId);
  res.json({ goal });
});

module.exports = router;
