**logger.js**: A file containing a utility function for logging messages.
```javascript
import winston from 'winston';

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.colorize(),
    winston.format.simple()
  ),
  transports: [
    new winston.transports.Console({
      level: 'info',
      handleExceptions: true
    }),
    new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
      handleExceptions: true
    }),
    new winston.transports.File({
      filename: 'logs/debug.log',
      level: 'debug',
      handleExceptions: true
    })
  ],
  exceptionHandlers: [
    new winston.transports.File({
      filename: 'logs/exceptions.log'
    })
  ]
});

logger.stream = {
  write: (message) => {
    logger.info(message);
  }
};

export default logger;
