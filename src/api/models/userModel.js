const mongoose = require('mongoose');
const bcrypt = require('bcrypt');

const userSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        minlength: 3,
        maxlength: 30
    },
    email: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        lowercase: true,
        match: [/\S+@\S+\.\S+/, 'Invalid email format']
    },
    password: {
        type: String,
        required: true,
        minlength: 8
    },
    createdAt: {
        type: Date,
        default: Date.now,
        immutable: true
    },
    isAdmin: {
        type: Boolean,
        default: false
    },
    walletAddress: {
        type: String,
        unique: true,
        sparse: true // Allows null values without violating uniqueness
    }
});

// Password hashing middleware
userSchema.pre('save', async function (next) {
    if (!this.isModified('password')) return next();
    try {
        const salt = await bcrypt.genSalt(10);
        this.password = await bcrypt.hash(this.password, salt);
        next();
    } catch (err) {
        next(err);
    }
});

// Password comparison method
userSchema.methods.comparePassword = async function (candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
};

// Auto-generate wallet address hook (example integration with blockchain)
userSchema.post('save', async function (doc) {
    if (!doc.walletAddress) {
        // Call your blockchain service to generate a wallet address
        // doc.walletAddress = await BlockchainService.generateWallet(doc._id);
        // await doc.save();
    }
});

module.exports = mongoose.model('User', userSchema);
