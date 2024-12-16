// tieredRewards.js

class TieredRewards {
    constructor() {
        this.userActivity = {}; // Store user activity levels
        this.tiers = [
            { level: 1, minActivity: 0, reward: 10 },
            { level: 2, minActivity: 5, reward: 25 },
            { level: 3, minActivity: 10, reward: 50 },
        ];
    }

    // Log user activity
    logActivity(userId) {
        if (!this.userActivity[userId]) {
            this.userActivity[userId] = 0;
        }
        this.userActivity[userId]++;
        console.log(`User  ${userId} activity logged. Total activity: ${this.userActivity[userId]}`);
    }

    // Get the reward for a user based on their activity
    getReward(userId) {
        const activityLevel = this.userActivity[userId] || 0;
        const tier = this.tiers.slice().reverse().find(t => activityLevel >= t.minActivity);
        return tier ? tier.reward : 0;
    }

    // Get tier details for a user
    getTierDetails(userId) {
        const activityLevel = this.userActivity[userId] || 0;
        const tier = this.tiers.find(t => activityLevel >= t.minActivity);
        return tier ? tier : null;
    }
}

// Example usage
const tieredRewards = new TieredRewards();
tieredRewards.logActivity('user123');
tieredRewards.logActivity('user123');
tieredRewards.logActivity('user123');
tieredRewards.logActivity('user123');
tieredRewards.logActivity('user123'); // User has now reached level 2

const reward = tieredRewards.getReward('user123');
console.log(`Reward for user123: $${reward}`);

const tierDetails = tieredRewards.getTierDetails('user123');
console.log(`Tier details for user123:`, tierDetails);

export default TieredRewards;
