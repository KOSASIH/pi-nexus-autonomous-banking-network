// paymentSecurity.js

import crypto from 'crypto';
import validator from 'validator';

class PaymentSecurity {
    constructor(encryptionKey) {
        this.encryptionKey = encryptionKey; // Key for encryption/decryption
        this.algorithm = 'aes-256-cbc'; // Encryption algorithm
    }

    // Validate payment details
    validatePaymentDetails(paymentDetails) {
        const { cardNumber, expiryDate, cvv, amount } = paymentDetails;

        if (!validator.isCreditCard(cardNumber)) {
            throw new Error('Invalid credit card number.');
        }
        if (!validator.isDate(expiryDate)) {
            throw new Error('Invalid expiry date.');
        }
        if (!validator.isNumeric(cvv) || cvv.length !== 3) {
            throw new Error('Invalid CVV.');
        }
        if (!validator.isNumeric(amount) || amount <= 0) {
            throw new Error('Invalid payment amount.');
        }

        console.log('Payment details validated successfully.');
    }

    // Encrypt sensitive payment data
    encryptData(data) {
        const iv = crypto.randomBytes(16); // Initialization vector
        const cipher = crypto.createCipheriv(this.algorithm, Buffer.from(this.encryptionKey), iv);
        let encrypted = cipher.update(data, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        return iv.toString('hex') + ':' + encrypted; // Return IV and encrypted data
    }

    // Decrypt sensitive payment data
    decryptData(encryptedData) {
        const [ivHex, encryptedText] = encryptedData.split(':');
        const iv = Buffer.from(ivHex, 'hex');
        const decipher = crypto.createDecipheriv(this.algorithm, Buffer.from(this.encryptionKey), iv);
        let decrypted = decipher.update(encryptedText, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        return decrypted;
    }

    // Process payment securely
    async processPayment(paymentDetails) {
        try {
            // Validate payment details
            this.validatePaymentDetails(paymentDetails);

            // Encrypt sensitive data
            const encryptedCardNumber = this.encryptData(paymentDetails.cardNumber);
            const encryptedCVV = this.encryptData(paymentDetails.cvv);

            // Simulate payment processing (e.g., sending to payment gateway)
            console.log('Processing payment...');
            // Here you would integrate with a payment gateway API

            // For demonstration, we just log the encrypted data
            console.log('Encrypted Card Number:', encryptedCardNumber);
            console.log('Encrypted CVV:', encryptedCVV);
            console.log('Payment processed successfully for amount:', paymentDetails.amount);
        } catch (error) {
            console.error('Error processing payment:', error.message);
        }
    }

    // Example usage
    exampleUsage() {
        const paymentDetails = {
            cardNumber: '4111111111111111', // Example card number (Visa)
            expiryDate: '12/25', // Example expiry date
            cvv: '123', // Example CVV
            amount: 100.00 // Example amount
        };

        this.processPayment(paymentDetails);
    }
}

// Example usage
const encryptionKey = crypto.randomBytes(32).toString('hex'); // Generate a random encryption key
const paymentSecurity = new PaymentSecurity(encryptionKey);
paymentSecurity.exampleUsage();

export default PaymentSecurity;
