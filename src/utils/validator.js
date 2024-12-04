// utils/validator.js

// Input validation utility
class Validator {
    static isEmail(email) {
        const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return regex.test(email);
    }

    static isNotEmpty(value) {
        return value && value.trim() !== '';
    }

    static isNumber(value) {
        return !isNaN(value) && !isNaN(parseFloat(value));
    }

    static validateUser Input(input) {
        const errors = [];
        if (!this.isNotEmpty(input.username)) {
            errors.push('Username is required.');
        }
        if (!this.isEmail(input.email)) {
            errors.push('Invalid email format.');
        }
        if (!this.isNumber(input.age)) {
            errors.push('Age must be a number.');
        }
        return errors.length > 0 ? errors : null;
    }
}

module.exports = Validator;
