// loyalty/rewards.js

class Reward {
    constructor(name, pointsRequired) {
        this.name = name;
        this.pointsRequired = pointsRequired;
    }

    getDetails() {
        return {
            name: this.name,
            pointsRequired: this.pointsRequired,
        };
    }
}

class RewardsCatalog {
    constructor() {
        this.rewards = [];
    }

    addReward(reward) {
        this.rewards.push(reward);
    }

    getAvailableRewards() {
        return this.rewards.map(reward => reward.getDetails());
    }

    redeemReward(userId, loyaltyProgram, rewardName) {
        const reward = this.rewards.find(r => r.name === rewardName);
        if (!reward) {
            throw new Error('Reward not found.');
        }
        loyaltyProgram.redeemPoints(userId, reward.pointsRequired);
        return `Reward "${rewardName}" redeemed successfully!`;
    }
}

module.exports = { Reward, RewardsCatalog };
