// payments/billing.js
const moment = require('moment');

class Invoice {
    constructor(userId, amount, dueDate) {
        this.userId = userId;
        this.amount = amount;
        this.dueDate = dueDate;
        this.isPaid = false;
    }

    markAsPaid() {
        this.isPaid = true;
        console.log(`Invoice for user ${this.userId} has been marked as paid.`);
    }

    getDetails() {
        return {
            userId: this.userId,
            amount: this.amount,
            dueDate: this.dueDate,
            isPaid: this.isPaid,
        };
    }
}

class Billing {
    constructor() {
        this.invoices = [];
    }

    generateInvoice(userId, amount) {
        const dueDate = moment().add(30, 'days').toDate(); // 30 days from now
        const invoice = new Invoice(userId, amount, dueDate);
        this.invoices.push(invoice);
        console.log(`Invoice generated for user ${userId}: $${amount}, due on ${dueDate}`);
        return invoice;
    }

    getInvoice(userId) {
        return this.invoices.filter(invoice => invoice.userId === userId);
    }

    processPayment(userId, amount) {
        const invoice = this.invoices.find(inv => inv.userId === userId && inv.amount === amount && !inv.isPaid);
        if (!invoice) {
            throw new Error('No unpaid invoice found for this user and amount.');
        }
        invoice.markAsPaid();
    }
}

module.exports = Billing;
