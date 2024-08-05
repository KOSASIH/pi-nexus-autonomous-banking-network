const { ErrorTypes } = require('./error-types');

class ErrorHandler {
  static handle(error) {
    if (error instanceof Error) {
      console.error(`Error: ${error.message}`);
      console.error(`Stack: ${error.stack}`);
    } else {
      console.error(`Unknown error: ${error}`);
    }

    switch (error.code) {
      case ErrorTypes.INVALID_REQUEST:
        return { statusCode: 400, message: 'Invalid request' };
      case ErrorTypes.UNAUTHORIZED:
        return { statusCode: 401, message: 'Unauthorized' };
      case ErrorTypes.FORBIDDEN:
        return { statusCode: 403, message: 'Forbidden' };
      case ErrorTypes.NOT_FOUND:
        return { statusCode: 404, message: 'Not found' };
      case ErrorTypes.INTERNAL_SERVER_ERROR:
        return { statusCode: 500, message: 'Internal server error' };
      default:
        return { statusCode: 500, message: 'Unknown error' };
    }
  }
}

module.exports = ErrorHandler;
