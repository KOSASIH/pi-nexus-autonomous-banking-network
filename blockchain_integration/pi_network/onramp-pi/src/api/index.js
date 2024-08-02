const express = require('express');
const app = express();
const Joi = require('joi');
const auth = require('./auth');
const rateLimit = require('express-rate-limit');

// Set up error handling
const { AuthenticationError, AuthorizationError } = require('./errors');

// Set up rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  delayMs: 0 // disable delaying - full speed until the max limit is reached
});

// Apply rate limiting to all routes
app.use(limiter);

// Set up authentication middleware
app.use(auth.authenticate);

// Define routes
app.use('/api/auth', require('./auth/routes'));
app.use('/api/users', require('./users/routes'));
app.use('/api/transactions', require('./transactions/routes'));

// Add advanced features for routing
app.use('/api/advanced', require('./advanced/routes'));

// Error handling
app.use((err, req, res, next) => {
  if (err instanceof AuthenticationError) {
    res.status(401).send({ message: err.message });
  } else if (err instanceof AuthorizationError) {
    res.status(403).send({ message: err.message });
  } else {
    res.status(500).send({ message: 'Internal Server Error' });
  }
});

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
