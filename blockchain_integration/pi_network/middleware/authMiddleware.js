// middleware/authMiddleware.js

const jwt = require('jsonwebtoken');

const SECRET_KEY = 'YOUR_SECRET_KEY'; // Replace with your actual secret key

const authMiddleware = (req, res, next) => {
    const token = req.headers['authorization']?.split(' ')[1]; // Bearer <token>

    if (!token) {
        return res.status(401).json({ message: 'No token provided, authorization denied.' });
    }

    jwt.verify(token, SECRET_KEY, (err, decoded) => {
        if (err) {
            return res.status(403).json({ message: 'Token is not valid.' });
        }
        req.user = decoded; // Attach user info to request object
        next(); // Proceed to the next middleware or route handler
    });
};

module.exports = authMiddleware;
