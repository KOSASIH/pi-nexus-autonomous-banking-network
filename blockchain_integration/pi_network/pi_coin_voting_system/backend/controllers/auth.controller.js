const express = require('express');
const router = express.Router();
const User = require('../models/user.model');
const passport = require('passport');

router.post('/register', async (req, res) => {
  try {
    const { username, password, email } = req.body;
    const user = new User({ username, password, email });
    await user.save();
    res.json({ message: 'User created successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

router.post(
  '/login',
  passport.authenticate('local', { session: false }),
  (req, res) => {
    const token = req.user.generateToken();
    res.json({ token });
  },
);

module.exports = router;
