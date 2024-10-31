// api/index.js

const express = require('express');
const userRoutes = require('./routes/userRoutes');
const contractRoutes = require('./routes/contractRoutes');
const errorHandlingMiddleware = require('../middleware/errorHandling');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json()); // Parse JSON request bodies

// API Routes
app.use('/api/users', userRoutes);
app.use('/api/contracts', contractRoutes);

// Error handling middleware
app.use(errorHandlingMiddleware);

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
