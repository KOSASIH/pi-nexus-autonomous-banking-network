const express = require('express');
const router = express.Router();
const User = require('../models/user.model');

router.post('/castVote', async (req, res) => {
  const { username, vote } = req.body;
  try {
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(401).json({ error: 'Invalid username' });
    }
    user.votes.push(vote);
    await user.save();
    res.json({ message: 'Vote cast successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

router.get('/getVotes', async (req, res) => {
  try {
    const users = await User.find().select('votes');
    const votes = users.reduce((acc, user) => acc.concat(user.votes), []);
    const voteCount = votes.length;
    const voteAverage = votes.reduce((a, b) => a + b, 0) / voteCount;
    const voteStandardDeviation = Math.sqrt(votes.reduce((a, b) => a + Math.pow(b - voteAverage, 2), 0) / voteCount);
    res.json({ voteCount, voteAverage, voteStandardDeviation });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

module.exports = router;
