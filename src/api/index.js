const express = require('express');
const mongoose = require('mongoose');
const userRoutes = require('./routes/userRoutes');
const transactionRoutes = require('./routes/transactionRoutes');
const contractRoutes = require('./routes/contractRoutes');
const authMiddleware = require('./middleware/authMiddleware');
const loggerMiddleware = require('./middleware/loggerMiddleware');
const errorHandler = require('./middleware/errorHandler');
const rateLimit = require('./middleware/rateLimitMiddleware');
const validateUser  = require('./middleware/validationMiddleware');
const swaggerSetup = require('./swagger');
const { connectDB } = require('./config/dbConfig');
const { PORT } = require('./config/serverConfig');
const WebSocket = require('./websocket'); // Import WebSocket setup

const app = express();

// Connect to the database
connectDB();

// Middleware
app.use(express.json());
app.use(loggerMiddleware); // Logging middleware
app.use(rateLimit); // Rate limiting middleware
app.use(authMiddleware); // Authentication middleware

// Swagger documentation
swaggerSetup(app);

// API Routes
app.use('/api/users', userRoutes);
app.use('/api/transactions', transactionRoutes);
app.use('/api/contracts', contractRoutes);

// Error handling middleware
app.use(errorHandler);

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

// Start WebSocket server
WebSocket.start(); // Start the WebSocket server
