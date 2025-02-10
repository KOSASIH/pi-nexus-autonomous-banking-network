const jwt = require('jsonwebtoken');
const { JWT_SECRET } = require('../config/serverConfig');

const authMiddleware = (req, res, next) => {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    if (!token) {
        return res.status(401).json({ success: false, message: 'Access denied' });
    }
    try {
        const verified = jwt.verify(token, JWT_SECRET);
        req.user = verified;
        next();
    } catch (error) {
        res.status(400).json({ success: false, message: 'Invalid token' });
    }
};

module.exports = authMiddleware;
