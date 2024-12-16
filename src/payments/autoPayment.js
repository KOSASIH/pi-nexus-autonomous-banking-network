// payments/autoPayment.js
const moment = require('moment');

class AutoPayment {
    constructor(userId, amount, frequency) {
        this.userId = userId;
        this.amount = amount;
        this.frequency = frequency; // e.g., 'daily', 'weekly', 'monthly'
        this.nextPaymentDate = this.calculateNextPaymentDate();
        this.isActive = true; // Indicates if the auto payment is active
    }

    calculateNextPaymentDate() {
        return moment().add(1, this.frequency).toDate();
    }

    processPayment() {
        if (!this.isActive) {
            throw new Error('Auto payment is not active.');
        }

        // Simulate payment processing (e.g., call to payment gateway)
        console.log(`Processing payment of $${this.amount} for user ${this.userId}`);
        
        // After processing, schedule the next payment
        this.nextPaymentDate = this.calculateNextPaymentDate();
        return this.nextPaymentDate;
    }

    cancelPayment() {
        this.isActive = false;
        console.log(`Auto payment for user ${this.userId} has been canceled.`);
    }

    resumePayment() {
        this.isActive = true;
        console.log(`Auto payment for user ${this.userId} has been resumed.`);
    }
}

module.exports = AutoPayment;
