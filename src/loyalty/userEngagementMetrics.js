// userEngagementMetrics.js

const fs = require('fs');
const path = require('path');

// Define the path for storing engagement metrics
const METRICS_FILE_PATH = path.join(__dirname, 'userEngagementMetrics.json');

// Initialize metrics storage
let metricsStorage = {};

// Load existing metrics from file
const loadMetrics = () => {
    if (fs.existsSync(METRICS_FILE_PATH)) {
        const data = fs.readFileSync(METRICS_FILE_PATH);
        metricsStorage = JSON.parse(data);
    }
};

// Save metrics to file
const saveMetrics = () => {
    fs.writeFileSync(METRICS_FILE_PATH, JSON.stringify(metricsStorage, null, 2));
};

// User Engagement Metrics Class
class UserEngagementMetrics {
    constructor(userId) {
        this.userId = userId;
        this.initializeUser Metrics();
    }

    // Initialize user metrics if not already present
    initializeUser Metrics() {
        if (!metricsStorage[this.userId]) {
            metricsStorage[this.userId] = {
                logins: 0,
                sessionDurations: [],
                featureUsage: {},
                lastLogin: null,
                totalEngagementTime: 0,
            };
            saveMetrics();
        }
    }

    // Track user login
    trackLogin() {
        const userMetrics = metricsStorage[this.userId];
        userMetrics.logins += 1;
        userMetrics.lastLogin = new Date().toISOString();
        saveMetrics();
    }

    // Track session duration
    trackSessionDuration(duration) {
        const userMetrics = metricsStorage[this.userId];
        userMetrics.sessionDurations.push(duration);
        userMetrics.totalEngagementTime += duration;
        saveMetrics();
    }

    // Track feature usage
    trackFeatureUsage(featureName) {
        const userMetrics = metricsStorage[this.userId];
        if (!userMetrics.featureUsage[featureName]) {
            userMetrics.featureUsage[featureName] = 0;
        }
        userMetrics.featureUsage[featureName] += 1;
        saveMetrics();
    }

    // Get user engagement metrics
    getUser Metrics() {
        return metricsStorage[this.userId];
    }

    // Generate a report of user engagement metrics
    generateEngagementReport() {
        const report = this.getUser Metrics();
        console.log('User  Engagement Report:', JSON.stringify(report, null, 2));
        return report;
    }
}

// Example usage
(async () => {
    const userId = 'user123';
    const engagementMetrics = new UserEngagementMetrics(userId);

    // Simulate user actions
    engagementMetrics.trackLogin();
    engagementMetrics.trackSessionDuration(120); // Session duration in seconds
    engagementMetrics.trackFeatureUsage('viewPortfolio');
    engagementMetrics.trackFeatureUsage('makeTransaction');
    engagementMetrics.trackSessionDuration(300); // Another session duration

    // Generate report
    engagementMetrics.generateEngagementReport();
})();

module.exports = UserEngagementMetrics;
