// twoFactorAuth.js

import speakeasy from 'speakeasy'; // Library for generating TOTP
import nodemailer from 'nodemailer'; // Library for sending emails
import twilio from 'twilio'; // Library for sending SMS

class TwoFactorAuth {
    constructor() {
        this.secret = null;
        this.userEmail = null;
        this.userPhone = null;
    }

    // Generate a new secret for the user
    generateSecret() {
        this.secret = speakeasy.generateSecret({ length: 20 });
        return this.secret.base32; // Return the base32 encoded secret
    }

    // Verify the TOTP code entered by the user
    verifyToken(token) {
        const verified = speakeasy.totp.verify({
            secret: this.secret.base32,
            encoding: 'base32',
            token: token,
            window: 1 // Allow a 1-minute window for token verification
        });
        return verified;
    }

    // Send the verification code via email
    async sendEmailVerification(email) {
        const transporter = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                user: process.env.EMAIL_USER, // Your email
                pass: process.env.EMAIL_PASS  // Your email password
            }
        });

        const token = speakeasy.totp({ secret: this.secret.base32, encoding: 'base32' });
        const mailOptions = {
            from: process.env.EMAIL_USER,
            to: email,
            subject: 'Your Verification Code',
            text: `Your verification code is: ${token}`
        };

        await transporter.sendMail(mailOptions);
        return token; // Return the token for testing purposes
    }

    // Send the verification code via SMS
    async sendSMSVerification(phone) {
        const client = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH_TOKEN);
        const token = speakeasy.totp({ secret: this.secret.base32, encoding: 'base32' });

        await client.messages.create({
            body: `Your verification code is: ${token}`,
            from: process.env.TWILIO_PHONE_NUMBER,
            to: phone
        });

        return token; // Return the token for testing purposes
    }

    // Set user contact information
    setUser Contact(email, phone) {
        this.userEmail = email;
        this.userPhone = phone;
    }
}

// Example usage
const twoFactorAuth = new TwoFactorAuth();
const secret = twoFactorAuth.generateSecret();
console.log('Secret:', secret);

twoFactorAuth.setUser Contact('user@example.com', '+1234567890');
twoFactorAuth.sendEmailVerification('user@example.com');
twoFactorAuth.sendSMSVerification('+1234567890');

export default TwoFactorAuth;
