// utils/logger.js

const fs = require('fs');
const path = require('path');

// Define log file path
const LOG_FILE_PATH = path.join(__dirname, 'app.log');

// Logger class
class Logger {
    static log(message) {
        const timestamp = new Date().toISOString();
        const logMessage = `${timestamp} - INFO: ${message}\n`;
        fs.appendFileSync(LOG_FILE_PATH, logMessage);
        console.log(logMessage);
    }

    static error(message) {
        const timestamp = new Date().toISOString();
        const logMessage = `${timestamp} - ERROR: ${message}\n`;
        fs.appendFileSync(LOG_FILE_PATH, logMessage);
        console.error(logMessage);
    }

    static warn(message) {
        const timestamp = new Date().toISOString();
        const logMessage = `${timestamp} - WARN: ${message}\n`;
        fs.appendFileSync(LOG_FILE_PATH, logMessage);
        console.warn(logMessage);
    }
}

module.exports = Logger;
