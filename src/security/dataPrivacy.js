// dataPrivacy.js

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Utility function to generate a random string for anonymization
function generateRandomString(length) {
    return crypto.randomBytes(length).toString('hex');
}

// Anonymize user data
function anonymizeData(userData) {
    return {
        ...userData,
        name: generateRandomString(10),
        email: `${generateRandomString(5)}@example.com`,
        phone: generateRandomString(10),
    };
}

// Store user consent
function storeUser Consent(userId, consent) {
    const consentData = {
        userId,
        consent,
        timestamp: new Date().toISOString(),
    };
    fs.appendFileSync(path.join(__dirname, 'consentLog.json'), JSON.stringify(consentData) + '\n');
}

// Retrieve user consent
function getUser Consent(userId) {
    const consentLog = fs.readFileSync(path.join(__dirname, 'consentLog.json'), 'utf-8');
    const consentEntries = consentLog.split('\n').filter(Boolean).map(JSON.parse);
    return consentEntries.filter(entry => entry.userId === userId);
}

// Handle data access requests
function handleDataAccessRequest(userId) {
    // Simulate fetching user data from a database
    const userData = {
        userId,
        name: 'John Doe',
        email: 'john.doe@example.com',
        phone: '1234567890',
    };

    return {
        userData: anonymizeData(userData),
        consent: getUser Consent(userId),
    };
}

// Exporting functions for use in other modules
module.exports = {
    anonymizeData,
    storeUser Consent,
    getUser Consent,
    handleDataAccessRequest,
};
