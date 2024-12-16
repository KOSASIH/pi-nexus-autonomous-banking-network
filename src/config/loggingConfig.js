// config/loggingConfig.js

const environment = require('./environment');

const loggingConfig = {
    logLevel: process.env.LOG_LEVEL || 'info', // Options: 'info', 'warn', 'error', 'debug'
    logFilePath: process.env.LOG_FILE_PATH || './logs/app.log',
};

module.exports = loggingConfig;
