const express = require('express');
const { createUser, loginUser, getUser } = require('../services/userService');
const authMiddleware = require('../middleware/authMiddleware');
const validateUser = require('../middleware/validationMiddleware');

const router = express.Router();

// Route to create a new user
router.post('/', validateUser, createUser);

// Route for user login
router.post('/login', loginUser);

// Route to get user details
router.get('/me', authMiddleware, getUser);

module.exports = router;
