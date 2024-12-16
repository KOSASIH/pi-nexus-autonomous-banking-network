// utils/errorHandler.js

// Error handling utility
class ErrorHandler {
    static handleError(err, req, res) {
        console.error(err);
        res.status(err.status || 500).json({
            success: false,
            message: err.message || 'Internal Server Error',
        });
    }
}

module.exports = ErrorHandler;
