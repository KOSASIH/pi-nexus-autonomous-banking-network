const express = require('express');
const router = express.Router();
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const Joi = require('joi');
const rateLimit = require('express-rate-limit');

// Import custom error classes
const { AuthenticationError, AuthorizationError } = require('./errors');

// Set up MongoDB connection
mongoose.connect('mongodb://localhost/pi-nexus-autonomous-banking-network', { useNewUrlParser: true, useUnifiedTopology: true });

// Define the User model
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  role: { type: String, enum: ['user', 'admin'], default: 'user' }
});

const User = mongoose.model('User', userSchema);

// Set up rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  delayMs: 0 // disable delaying - full speed until the max limit is reached
});

// Apply rate limiting to all routes
router.use(limiter);

// Define the authentication routes
router.post('/register', async (req, res, next) => {
  try {
    const { email, password } = req.body;
    const user = new User({ email, password });
    const salt = await bcrypt.genSalt(10);
    user.password = await bcrypt.hash(password, salt);
    await user.save();
    res.status(201).send({ message: 'User created successfully' });
  } catch (err) {
    next(err);
  }
});

router.post('/login', async (req, res, next) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email });
    if (!user) {
      throw new AuthenticationError('Invalid email or password');
    }
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      throw new AuthenticationError('Invalid email or password');
    }
    const token = jwt.sign({ userId: user._id, role: user.role }, process.env.SECRET_KEY, { expiresIn: '1h' });
    res.status(200).send({ token });
  } catch (err) {
    next(err);
  }
});

router.post('/refresh-token', async (req, res, next) => {
  try {
    const { token } = req.body;
    const decoded = jwt.verify(token, process.env.SECRET_KEY);
    const user = await User.findById(decoded.userId);
    if (!user) {
      throw new AuthenticationError('Invalid token');
    }
    const newToken = jwt.sign({ userId: user._id, role: user.role }, process.env.SECRET_KEY, { expiresIn: '1h' });
    res.status(200).send({ token: newToken });
  } catch (err) {
    next(err);
  }
});

// Define the authentication middleware
const authenticate = async (req, res, next) => {
  try {
    const token = req.header('Authorization').replace('Bearer ', '');
    const decoded = jwt.verify(token, process.env.SECRET_KEY);
    req.user = await User.findById(decoded.userId);
    next();
  } catch (err) {
    next(new AuthenticationError('Invalid token'));
  }
};

// Apply authentication middleware to protected routes
router.use('/protected', authenticate);

router.get('/protected/me', async (req, res) => {
  res.status(200).send({ user: req.user });
});

// Error handling
router.use((err, req, res, next) => {
  if (err instanceof AuthenticationError) {
    res.status(401).send({ message: err.message });
  } else if (err instanceof AuthorizationError) {
    res.status(403).send({ message: err.message });
  } else {
    res.status(500).send({ message: 'Internal Server Error' });
  }
});

module.exports = router;
