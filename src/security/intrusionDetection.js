// intrusionDetection.js

import fs from 'fs';
import path from 'path';

class IntrusionDetection {
    constructor() {
        this.logFilePath = path.join(__dirname, 'intrusionLogs.json');
        this.suspiciousActivities = [];
    }

    // Log suspicious activity
    logSuspiciousActivity(activity) {
        const timestamp = new Date().toISOString();
        const logEntry = { timestamp, activity };
        this.suspiciousActivities.push(logEntry);
        this.saveLogs();
    }

    // Save logs to a file
    saveLogs() {
        fs.writeFileSync(this.logFilePath, JSON.stringify(this.suspiciousActivities, null, 2));
    }

    // Check for unauthorized access attempts
    monitorAccess(userId, ipAddress) {
        // Simulate checking for unauthorized access
        const unauthorizedAccess = this.checkUnauthorizedAccess(userId, ipAddress);
        if (unauthorizedAccess) {
            this.logSuspiciousActivity(`Unauthorized access attempt by user ${userId} from IP ${ipAddress}`);
            return true; // Indicate that an intrusion was detected
        }
        return false; // No intrusion detected
    }

    // Simulated method to check for unauthorized access
    checkUnauthorizedAccess(userId, ipAddress) {
        // In a real application, this would check against a database or security logs
        const unauthorizedIPs = ['192.168.1.100', '10.0.0.5']; // Example unauthorized IPs
        return unauthorizedIPs.includes(ipAddress);
    }

    // Retrieve logged suspicious activities
    getSuspiciousActivities```javascript
    getSuspiciousActivities() {
        return this.suspiciousActivities;
    }
}

// Example usage
const intrusionDetection = new IntrusionDetection();
const userId = 'user123';
const ipAddress = '192.168.1.100'; // Example IP address

if (intrusionDetection.monitorAccess(userId, ipAddress)) {
    console.log('Intrusion detected! Check logs for details.');
} else {
    console.log('Access granted. No intrusion detected.');
}

// Retrieve and display suspicious activities
const logs = intrusionDetection.getSuspiciousActivities();
console.log('Suspicious Activities:', logs);

export default IntrusionDetection;
