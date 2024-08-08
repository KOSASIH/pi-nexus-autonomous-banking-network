class Logger {
    log(message) {
        console.log(`[${new Date().toISOString()}] ${message}`);
    }

    error(message) {
        console.error(`[${new Date().toISOString()}] ${message}`);
    }
}

module.exports = Logger;
