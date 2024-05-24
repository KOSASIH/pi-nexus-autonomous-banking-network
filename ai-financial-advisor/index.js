const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const app = express();
const port = process.env.PORT || 5000;

app.use(bodyParser.json());
app.use(cors());

// Import routes
const authRoutes = require('./routes/auth');
const dashboardRoutes = require('./routes/dashboard');
const financialAdvisorRoutes = require('./routes/financialAdvisor');

// Use routes
app.use('/api/auth', authRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/financial-advisor', financialAdvisorRoutes);

// Start server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
