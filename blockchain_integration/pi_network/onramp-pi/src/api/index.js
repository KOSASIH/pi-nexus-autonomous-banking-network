const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const cors = require('cors');
const transactionRoutes = require('./transactions');

app.use(cors());
app.use(bodyParser.json());
app.use('/api/transactions', transactionRoutes);

module.exports = app;
