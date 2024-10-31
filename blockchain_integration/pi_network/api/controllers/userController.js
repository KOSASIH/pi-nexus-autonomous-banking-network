// api/controllers/userController.js

const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

const users = []; // In-memory user storage (replace with a database in production)
const SECRET_KEY = 'YOUR_SECRET_KEY'; // Replace with your actual secret key

const userController = {
    registerUser: async (req, res) => {
        const { username, password } = req.body;

        // Check if user already exists
        const existingUser = users.find(user => user.username === username);
        if (existingUser) {
            return res.status(400).json({ message: 'User already exists.' });
        }

        // Hash the password
        const hashedPassword = await bcrypt.hash(password, 10);
        const newUser = { username, password: hashedPassword };
        users.push(newUser); // Save user (replace with database logic)

        res.status(201).json({ message: 'User registered successfully.' });
    },

    loginUser: async (req, res) => {
        const { username, password } = req.body;
        const user = users.find(user => user.username === username);

        if (!user || !(await bcrypt.compare(password, user.password))) {
            return res.status(401).json({ message: 'Invalid credentials.' });
        }

        // Generate JWT
        const token = jwt.sign({ username }, SECRET_KEY, { expiresIn: '1h' });
        res.json({ token });
    },

    getUserProfile: (req, res) => {
        // Assuming user is authenticated and user info is attached to req.user
        res.json({ message: 'User profile data', user: req.user });
    },
};

module.exports = userController;
