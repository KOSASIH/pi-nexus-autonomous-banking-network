const express = require('express');
const app = express();
const mongoose = require('mongoose');

// Connect to MongoDB
mongoose.connect('mongodb://localhost/sidra-dex', { useNewUrlParser: true, useUnifiedTopology: true });

// Define API routes
app.use('/api/users', require('./routes/users'));
app.use('/api/tokens', require('./routes/tokens'));

// Start server
const port = 3000;
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
