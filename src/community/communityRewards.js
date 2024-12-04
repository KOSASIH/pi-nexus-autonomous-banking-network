// communityRewards.js

class CommunityRewards {
    constructor() {
        this.users = {}; // Object to store user data
        this.rewards = []; // Array to store available rewards
    }

    // Add a new user
    addUser (userId) {
        if (!this.users[userId]) {
            this.users[userId] = {
                engagementPoints: 0,
                rewardsEarned: [],
            };
            console.log(`User  ${userId} added.`);
        } else {
            console.log(`User  ${userId} already exists.`);
        }
    }

    // Add a new reward
    addReward(reward) {
        this.rewards.push(reward);
        console.log(`Reward "${reward.name}" added with ${reward.points} points.`);
    }

    // Update user engagement points
    updateEngagement(userId, points) {
        if (this.users[userId]) {
            this.users[userId].engagementPoints += points;
            console.log(`User  ${userId} engagement points updated to ${this.users[userId].engagementPoints}.`);
            this.checkRewards(userId);
        } else {
            console.log(`User  ${userId} does not exist.`);
        }
    }

    // Check if user qualifies for any rewards
    checkRewards(userId) {
        const user = this.users[userId];
        this.rewards.forEach((reward) => {
            if (user.engagementPoints >= reward.points && !user.rewardsEarned.includes(reward.name)) {
                user.rewardsEarned.push(reward.name);
                console.log(`User  ${userId} earned reward: "${reward.name}"`);
            }
        });
    }

    // Redeem a reward
    redeemReward(userId, rewardName) {
        const user = this.users[userId];
        if (user && user.rewardsEarned.includes(rewardName)) {
            user.rewardsEarned = user.rewardsEarned.filter(reward => reward !== rewardName);
            console.log(`User  ${userId} redeemed reward: "${rewardName}"`);
        } else {
            console.log(`User  ${userId} does not have reward: "${rewardName}"`);
        }
    }

    // Example usage
    exampleUsage() {
        // Adding users
        this.addUser ('user1');
        this.addUser ('user2');

        // Adding rewards
        this.addReward({ name: '10% Discount', points: 100 });
        this.addReward({ name: 'Free Merchandise', points: 200 });

        // Simulating engagement
        this.updateEngagement('user1', 50);
        this.updateEngagement('user1', 60); // Should earn the discount reward
        this.updateEngagement('user2', 150); // Should not earn any rewards yet
        this.updateEngagement('user2', 100); // Should earn the discount reward

        // Redeeming rewards
        this.redeemReward('user1', '10% Discount');
        this.redeemReward('user2', 'Free Merchandise'); // Should not be able to redeem
    }
}

// Example usage
const communityRewards = new CommunityRewards();
communityRewards.exampleUsage();

export default CommunityRewards;
