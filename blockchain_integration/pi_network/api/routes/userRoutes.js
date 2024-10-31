// api/routes/userRoutes.js

const express = require('express');
const userController = require('../controllers/userController');

const router = express.Router();

// User registration
router.post('/register', userController.registerUser );

// User login
router.post('/login', userController.loginUser );

// Get user profile
router.get('/profile', userController.getUser Profile);

module.exports = router;
