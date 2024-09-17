// input-validator.js
const validator = require('validator');

class InputValidator {
  validateString(input) {
    return validator.isString(input) && validator.trim(input) !== '';
  }

  validateEmail(input) {
    return validator.isEmail(input);
  }

  validatePassword(input) {
    return validator.isStrongPassword(input, {
      minLength: 12,
      minLowercase: 1,
      minUppercase: 1,
      minNumbers: 1,
      minSymbols: 1,
    });
  }
}

module.exports = InputValidator;
