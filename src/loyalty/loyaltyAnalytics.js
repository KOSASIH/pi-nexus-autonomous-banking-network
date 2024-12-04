// loyaltyAnalytics.js

class LoyaltyAnalytics {
    constructor() {
        this.userActivities = []; // Array to store user activities
        this.rewardRedemptions = []; // Array to store reward redemptions
    }

    // Log user activity
    logUser Activity(userId, activityType, pointsEarned) {
        const activity = {
            userId,
            activityType,
            pointsEarned,
            timestamp: new Date().toISOString()
        };
        this.userActivities.push(activity);
        console.log(`Logged activity for user ${userId}: ${activityType} with ${pointsEarned} points.`);
    }

    // Log reward redemption
    logRewardRedemption(userId, rewardId, pointsSpent) {
        const redemption = {
            userId,
            rewardId,
            pointsSpent,
            timestamp: new Date().toISOString()
        };
        this.rewardRedemptions.push(redemption);
        console.log(`Logged reward redemption for user ${userId}: Reward ID ${rewardId} with ${pointsSpent} points.`);
    }

    // Calculate total points earned by a user
    calculateTotalPoints(userId) {
        const totalPoints = this.userActivities
            .filter(activity => activity.userId === userId)
            .reduce((total, activity) => total + activity.pointsEarned, 0);
        return totalPoints;
    }

    // Calculate total rewards redeemed by a user
    calculateTotalRewardsRedeemed(userId) {
        const totalRedeemed = this.rewardRedemptions
            .filter(redemption => redemption.userId === userId)
            .reduce((total, redemption) => total + redemption.pointsSpent, 0);
        return totalRedeemed;
    }

    // Generate a report of user activities
    generateUser Report(userId) {
        const activities = this.userActivities.filter(activity => activity.userId === userId);
        const totalPoints = this.calculateTotalPoints(userId);
        const totalRedeemed = this.calculateTotalRewardsRedeemed(userId);

        return {
            userId,
            activities,
            totalPoints,
            totalRedeemed
        };
    }

    // Generate a summary report of the loyalty program
    generateSummaryReport() {
        const totalUsers = new Set(this.userActivities.map(activity => activity.userId)).size;
        const totalActivities = this.userActivities.length;
        const totalRedemptions = this.rewardRedemptions.length;

        return {
            totalUsers,
            totalActivities,
            totalRedemptions
        };
    }

    // Example usage
    exampleUsage() {
        // Log some user activities
        this.logUser Activity('user123', 'purchase', 100);
        this.logUser Activity('user123', 'review', 50);
        this.logUser Activity('user456', 'purchase', 200);

        // Log some reward redemptions
        this.logRewardRedemption('user123', 'reward1', 150);
        this.logRewardRedemption('user456', 'reward2', 100);

        // Generate user report
        const userReport = this.generateUser Report('user123');
        console.log('User  Report:', userReport);

        // Generate summary report
        const summaryReport = this.generateSummaryReport();
        console.log('Summary Report:', summaryReport);
    }
}

// Example usage
const loyaltyAnalytics = new LoyaltyAnalytics();
loyaltyAnalytics.exampleUsage();

export default LoyaltyAnalytics;
