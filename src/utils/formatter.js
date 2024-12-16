// utils/formatter.js

// Data formatting utility
class Formatter {
    static formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
        }).format(amount);
    }

    static formatDate(date) {
        return new Intl.DateTimeFormat('en-US').format(new Date(date));
    }

    static formatResponse(data) {
        return {
            success: true,
            data: data,
        };
    }
}

module.exports = Formatter;
