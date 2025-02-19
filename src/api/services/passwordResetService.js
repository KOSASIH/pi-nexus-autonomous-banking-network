const PasswordResetModel = require('../models/passwordResetModel');
const UserModel = require('../models/userModel');
const crypto = require('crypto');
const { sendEmail } = require('../utils/email'); // Assume you have an email utility

// Request a password reset
const requestPasswordReset = async (req, res) => {
    const { email } = req.body;
    const user = await UserModel.findOne({ email });
    if (!user) {
        return res.status(404).json({ success: false, message: 'User not found' });
    }

    const token = crypto.randomBytes(32).toString('hex');
    await PasswordResetModel.create({ userId: user._id, token });

    const resetLink = `http://localhost:3000/reset-password/${token}`;
    await sendEmail(user.email, 'Password Reset', `Click this link to reset your password: ${resetLink}`);

    res.json({ success: true, message: 'Password reset link sent to your email' });
};

// Reset the password
const resetPassword = async (req, res) => {
    const { token, newPassword } = req.body;
    const passwordReset = await PasswordResetModel.findOne({ token });
    if (!passwordReset) {
        return res.status(400).json({ success: false, message: 'Invalid or expired token' });
    }

    const user = await UserModel.findById(passwordReset.userId);
    user.password = newPassword; // Hashing should be done in the user model
    await user.save();
    await PasswordResetModel.deleteOne({ token });

    res.json({ success: true, message: 'Password has been reset' });
};

module.exports = {
    requestPasswordReset,
    resetPassword,
};
