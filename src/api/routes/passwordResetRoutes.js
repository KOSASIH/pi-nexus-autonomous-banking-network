const express = require('express');
const { requestPasswordReset, resetPassword } = require('../services/passwordResetService');

const router = express.Router();

// Route to request a password reset
router.post('/request', requestPasswordReset);

// Route to reset the password
router.post('/reset', resetPassword);

module.exports = router;
