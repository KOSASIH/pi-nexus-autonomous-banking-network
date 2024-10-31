// middleware/analyticsMiddleware.js

const analyticsMiddleware = (req, res, next) => {
    const start = Date.now(); // Record the start time

    res.on('finish', () => {
        const duration = Date.now() - start; // Calculate duration
        const logData = {
            method: req.method,
            url: req.originalUrl,
            status: res.statusCode,
            duration,
        };
        console.log('Request Analytics:', logData); // Log the analytics data
        // Here you can send the logData to an analytics service if needed
    });

    next(); // Proceed to the next middleware or route handler
};

module.exports = analyticsMiddleware;
