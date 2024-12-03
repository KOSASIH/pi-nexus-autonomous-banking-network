// src/security/auth.js
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const User = require('../models/User'); // Assuming you have a User model
require('dotenv').config();

const SALT_ROUNDS = 10;

async function registerUser (username, password) {
    const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
    const newUser  = new User({ username, password: hashedPassword });
    await newUser .save();
    return newUser ;
}

async function authenticateUser (username, password) {
    const user = await User.findOne({ username });
    if (!user) throw new Error('User  not found');

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) throw new Error('Invalid password');

    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
    return { token, user };
}

function verifyToken(token) {
    return jwt.verify(token, process.env.JWT_SECRET);
}

module.exports = { registerUser , authenticateUser , verifyToken };
