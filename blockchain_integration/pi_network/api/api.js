const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const cors = require('cors');

// Connect to MongoDB database
mongoose.connect('mongodb://localhost/pi-network-db', { useNewUrlParser: true, useUnifiedTopology: true });

// Define Pi Coin model
const PiCoin = mongoose.model('PiCoin', {
  value: { type: Number, default: 314.159 },
  votes: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Vote' }]
});

// Define Vote model
const Vote = mongoose.model('Vote', {
  value: { type: Number, default: 314.159 },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  option: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

// Define User model
const User = mongoose.model('User', {
  username: { type: String, required: true },
  password: { type: String, required: true },
  piCoins: [{ type: mongoose.Schema.Types.ObjectId, ref: 'PiCoin' }]
});

// Middlewares
app.use(bodyParser.json());
app.use(cors());

// API Endpoints

// Get Pi Coin value
app.get('/api/pi-coin/value', async (req, res) => {
  const piCoin = await PiCoin.findOne();
  res.json({ value: piCoin.value });
});

// Cast a vote
app.post('/api/vote', async (req, res) => {
  const { option } = req.body;
  const user = await User.findOne({ username: req.headers.username });
  const vote = new Vote({ userId: user._id, option });
  await vote.save();
  res.json({ message: 'Vote cast successfully' });
});

// Get user's Pi Coin balance
app.get('/api/user/pi-coins', async (req, res) => {
  const user = await User.findOne({ username: req.headers.username });
  const piCoins = await PiCoin.find({ _id: { $in: user.piCoins } });
  res.json({ piCoins: piCoins.map(piCoin => piCoin.value) });
});

// Start server
const port = 3000;
app.listen(port, () => {
  console.log(`API server listening on port ${port}`);
});
