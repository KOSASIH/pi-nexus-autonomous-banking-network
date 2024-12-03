// dataAnalyticsAPI.js

class DataAnalyticsAPI {
    constructor() {
        this.analyticsData = {
            userActivity: [],
            transactionStats: [],
        };
    }

    // Log user activity
    logUserActivity(userId, activity) {
        const activityEntry = {
            userId,
            activity,
            timestamp: new Date(),
        };
        this.analyticsData.userActivity.push(activityEntry);
        console.log(`User activity logged:`, activityEntry);
    }

    // Log transaction statistics
    logTransaction(transaction) {
        this.analyticsData.transactionStats.push(transaction);
        console.log(`Transaction logged:`, transaction);
    }

    // Get user activity data
    getUserActivity(userId) {
        return this.analyticsData.userActivity.filter(entry => entry.userId === userId);
    }

    // Get transaction statistics
    getTransactionStats() {
        return this.analyticsData.transactionStats;
    }

    // Get overall analytics summary
    getAnalyticsSummary() {
        return {
            totalUsers: new Set(this.analyticsData.userActivity.map(entry => entry.userId)).size,
            totalTransactions: this.analyticsData.transactionStats.length,
            totalActivityLogs: this.analyticsData.userActivity.length,
        };
    }
}

// Example usage
const analyticsAPI = new DataAnalyticsAPI();
analyticsAPI.logUserActivity('user123', 'Logged in');
analyticsAPI.logUserActivity('user123', 'Made a purchase');
analyticsAPI.logTransaction({ id: 'txn_123456', amount: 100, currency: 'USD', date: new Date() });

const userActivity = analyticsAPI.getUserActivity('user123');
console.log('User Activity for user123:', userActivity);

const transactionStats = analyticsAPI.getTransactionStats();
console.log('Transaction Statistics:', transactionStats);

const analyticsSummary = analyticsAPI.getAnalyticsSummary();
console.log('Analytics Summary:', analyticsSummary);

export default DataAnalyticsAPI;
