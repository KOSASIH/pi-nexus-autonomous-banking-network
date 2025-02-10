const express = require('express');
const { createUser , getUser , loginUser  } = require('../services/userService');

const router = express.Router();

// Route to create a new user
router.post('/', createUser );

// Route to get user details
router.get('/:id', getUser );

// Route to login a user
router.post('/login', loginUser );

module.exports = router;
