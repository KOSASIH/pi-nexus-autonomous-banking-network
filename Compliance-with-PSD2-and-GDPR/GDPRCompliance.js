// GDPRCompliance.js

class GDPRCompliance {
    constructor() {
        this.userData = new Map(); // Store user data
        this.consentRecords = new Map(); // Store user consent records
    }

    // Method to collect user data
    collectUser Data(userId, data) {
        if (!this.consentRecords.get(userId)) {
            throw new Error("User  consent not given.");
        }
        this.userData.set(userId, data);
        console.log(`User  data collected for ${userId}`);
    }

    // Method to record user consent
    recordConsent(userId, consent) {
        this.consentRecords.set(userId, consent);
        console.log(`Consent recorded for ${userId}: ${consent}`);
    }

    // Method to check user consent
    checkConsent(userId) {
        return this.consentRecords.get(userId) || false;
    }

    // Method to handle data access requests
    handleDataAccessRequest(userId) {
        if (!this.userData.has(userId)) {
            throw new Error("No data found for this user.");
        }
        return this.userData.get(userId);
    }

    // Method to handle data deletion requests
    handleDataDeletionRequest(userId) {
        if (this.userData.has(userId)) {
            this.userData.delete(userId);
            this.consentRecords.delete(userId);
            console.log(`Data deleted for ${userId}`);
        } else {
            throw new Error("No data found for this user.");
        }
    }

    // Method to log compliance actions
    logComplianceAction(action, userId) {
        console.log(`Compliance action: ${action} for user: ${userId}`);
    }
}

// Example usage
const gdpr = new GDPRCompliance();

// User consent management
const userId = 'user123';
gdpr.recordConsent(userId, true);

// Collect user data
try {
    gdpr.collectUser Data(userId, { name: "John Doe", email: "john.doe@example.com" });
} catch (error) {
    console.error(error.message);
}

// Handle data access request
try {
    const userData = gdpr.handleDataAccessRequest(userId);
    console.log("User  Data:", userData);
} catch (error) {
    console.error(error.message);
}

// Handle data deletion request
try {
    gdpr.handleDataDeletionRequest(userId);
} catch (error) {
    console.error(error.message);
                     }
