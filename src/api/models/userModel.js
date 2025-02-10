const mongoose = require('mongoose');
const bcrypt = require('bcrypt');

const userSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,
    },
    email: {
        type: String,
        required: true,
        unique: true,
    },
    password: {
        type: String,
        required: true,
    },
    role: {
        type: String,
        enum: ['user', 'admin'],
        default: 'user',
    },
    createdAt: {
        type: Date,
        default: Date.now,
    },
});

// Hash the password before saving
userSchema.pre('save', async function (next) {
    if (!this.isModified('password')) return next();
    this.password = await bcrypt.hash(this.password, 10);
    next();
});

// Method to compare passwords
userSchema.methods.comparePassword = async function (password) {
    return await bcrypt.compare(password, this.password);
};

const UserModel = mongoose.model('User', userSchema);
module.exports = UserModel;
