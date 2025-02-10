const mongoose = require('mongoose');

const passwordResetSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
    },
    token: {
        type: String,
        required: true,
    },
    createdAt: {
        type: Date,
        expires: '1h', // Token expires in 1 hour
        type: Date,
        default: Date.now,
    },
});

const PasswordResetModel = mongoose.model('PasswordReset', passwordResetSchema);
module.exports = PasswordResetModel;
