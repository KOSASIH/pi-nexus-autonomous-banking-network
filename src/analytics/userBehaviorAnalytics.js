// userBehaviorAnalytics.js

class UserBehaviorAnalytics {
    constructor() {
        this.userActivities = {}; // Store user activities by user ID
    }

    // Log user activity
    logActivity(userId, activity) {
        if (!this.userActivities[userId]) {
            this.userActivities[userId] = [];
        }
        const activityEntry = {
            activity,
            timestamp: new Date(),
        };
        this.userActivities[userId].push(activityEntry);
        console.log(`Activity logged for user ${userId}:`, activityEntry);
    }

    // Get user activity history
    getUser Activity(userId) {
        return this.userActivities[userId] || [];
    }

    // Analyze user behavior to provide personalized recommendations
    analyzeUser Behavior(userId) {
        const activities = this.getUser Activity(userId);
        const recommendations = [];

        // Example analysis: Recommend based on activity types
        const activityCounts = activities.reduce((acc, entry) => {
            acc[entry.activity] = (acc[entry.activity] || 0) + 1;
            return acc;
        }, {});

        for (const [activity, count] of Object.entries(activityCounts)) {
            if (count > 2) { // Example threshold for recommendations
                recommendations.push(`Consider exploring more about ${activity}.`);
            }
        }

        return recommendations.length > 0 ? recommendations : ['No recommendations available.'];
    }
}

// Example usage
const userAnalytics = new UserBehaviorAnalytics();
userAnalytics.logActivity('user123', 'viewed_product');
userAnalytics.logActivity('user123', 'added_to_cart');
userAnalytics.logActivity('user123', 'viewed_product');
userAnalytics.logActivity('user123', 'completed_purchase');

const userActivity = userAnalytics.getUser Activity('user123');
console.log('User  Activity for user123:', userActivity);

const recommendations = userAnalytics.analyzeUser Behavior('user123');
console.log('Personalized Recommendations for user123:', recommendations);

export default UserBehaviorAnalytics;
