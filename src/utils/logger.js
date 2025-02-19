import fs from 'fs';
import path from 'path';

const logFilePath = path.join(__dirname, 'error.log');

const logError = (error) => {
    const timestamp = new Date().toISOString();
    const logMessage = `${timestamp} - ERROR: ${error}\n`;
    fs.appendFileSync(logFilePath, logMessage);
};

export default logError;
