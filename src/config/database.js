// config/database.js

const environment = require('./environment');

const databaseConfig = {
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_PORT || 5432,
    user: process.env.DB_USER || 'user',
    password: process.env.DB_PASSWORD || 'password',
    database: process.env.DB_NAME || 'mydatabase',
    dialect: 'postgres', // Change this based on your database (e.g., 'mysql', 'sqlite', etc.)
};

module.exports = databaseConfig;
