// config/environment.js

const dotenv = require('dotenv');

// Load environment variables from a .env file into process.env
dotenv.config();

const environment = {
    NODE_ENV: process.env.NODE_ENV || 'development',
    PORT: process.env.PORT || 3000,
    API_URL: process.env.API_URL || 'http://localhost:3000/api',
};

module.exports = environment;
