const mongoose = require('mongoose');

const credentialSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User ',
        required: true,
    },
    credentialType: {
        type: String,
        required: true,
    },
    credentialData: {
        type: Object,
        required: true,
    },
    issuedAt: {
        type: Date,
        default: Date.now,
    },
    expiresAt: {
        type: Date,
    },
    status: {
        type: String,
        enum: ['active', 'revoked'],
        default: 'active',
    },
});

// Method to revoke a credential
credentialSchema.methods.revoke = function () {
    this.status = 'revoked';
    return this.save();
};

const Credential = mongoose.model('Credential', credentialSchema);
module.exports = Credential;
