// compliance.js

const fs = require('fs');
const path = require('path');

// Compliance regulations
const regulations = {
    GDPR: {
        description: 'General Data Protection Regulation',
        requirements: [
            'User  consent must be obtained before data collection.',
            'Users have the right to access their data.',
            'Data must be securely stored and processed.',
        ],
    },
    CCPA: {
        description: 'California Consumer Privacy Act',
        requirements: [
            'Consumers have the right to know what personal data is collected.',
            'Consumers can request deletion of their personal data.',
            'Businesses must provide a clear privacy policy.',
        ],
    },
};

// Log compliance checks
function logComplianceCheck(userId, regulation, status) {
    const logEntry = {
        userId,
        regulation,
        status,
        timestamp: new Date().toISOString(),
    };
    fs.appendFileSync(path.join(__dirname, 'complianceLog.json'), JSON.stringify(logEntry) + '\n');
}

// Check compliance for a specific regulation
function checkCompliance(userId, regulation) {
    const complianceStatus = regulations[regulation] ? 'Compliant' : 'Non-Compliant';
    logComplianceCheck(userId, regulation, complianceStatus);
    return {
        regulation,
        status: complianceStatus,
        requirements: regulations[regulation]?.requirements || [],
    };
}

// Generate compliance report
function generateComplianceReport() {
    const report = [];
    const complianceLog = fs.readFileSync(path.join(__dirname, 'complianceLog.json'), 'utf-8');
    const logEntries = complianceLog.split('\n').filter(Boolean).map(JSON.parse);
    
    logEntries.forEach(entry => {
        report.push({
            userId: entry.userId,
            regulation: entry.regulation,
            status: entry.status,
            timestamp: entry.timestamp,
        });
    });

    return report;
}

// Exporting functions for use in other modules
module.exports = {
    checkCompliance,
    generateComplianceReport,
};
