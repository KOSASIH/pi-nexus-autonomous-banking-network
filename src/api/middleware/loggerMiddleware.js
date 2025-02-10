const logger = require('../utils/logger');

const loggerMiddleware = (req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - start;
        logger.info(`${req.method} ${req.url} ${res.statusCode} - ${duration}ms`);
    });
    next();
};

module.exports = loggerMiddleware;
