// referralProgram.js

class ReferralProgram {
    constructor() {
        this.referrals = {}; // Store referrals by user ID
        this.rewards = {};   // Store rewards by user ID
    }

    // Refer a new user
    refer(referrerId, referredId) {
        if (!this.referrals[referrerId]) {
            this.referrals[referrerId] = [];
        }
        this.referrals[referrerId].push(referredId);
        console.log(`User  ${referrerId} referred ${referredId}.`);
    }

    // Reward the referrer for a successful referral
    rewardReferrer(referrerId) {
        if (!this.rewards[referrerId]) {
            this.rewards[referrerId] = 0;
        }
        this.rewards[referrerId] += 50; // Example reward amount
        console.log(`User  ${referrerId} has been rewarded. Total rewards: $${this.rewards[referrerId]}`);
    }

    // Get referral details for a user
    getReferralDetails(userId) {
        return {
            referrals: this.referrals[userId] || [],
            rewards: this.rewards[userId] || 0,
        };
    }
}

// Example usage
const referralProgram = new ReferralProgram();
referralProgram.refer('user123', 'user456');
referralProgram.rewardReferrer('user123');

const referralDetails = referralProgram.getReferralDetails('user123');
console.log(`Referral details for user123:`, referralDetails);

export default ReferralProgram;
