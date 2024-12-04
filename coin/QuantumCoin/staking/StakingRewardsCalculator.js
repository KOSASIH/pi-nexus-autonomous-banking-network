// StakingRewardsCalculator.js

class StakingRewardsCalculator {
    constructor() {
        this.apy = 0; // Annual Percentage Yield
    }

    // Set the annual percentage yield
    setAPY(apy) {
        if (apy < 0) {
            throw new Error('APY must be a non-negative value.');
        }
        this.apy = apy;
    }

    // Calculate rewards based on the amount staked and duration
    calculateRewards(amountStaked, durationInDays) {
        if (amountStaked <= 0) {
            throw new Error('Amount staked must be greater than zero.');
        }
        if (durationInDays <= 0) {
            throw new Error('Duration must be greater than zero.');
        }

        // Calculate daily interest rate
        const dailyRate = this.apy / 365 / 100;

        // Calculate total rewards
        const totalRewards = amountStaked * dailyRate * durationInDays;
        return totalRewards;
    }

    // Calculate total amount after staking
    calculateTotalAmount(amountStaked, durationInDays) {
        const rewards = this.calculateRewards(amountStaked, durationInDays);
        return amountStaked + rewards;
    }
}

// Example usage
const calculator = new StakingRewardsCalculator();
calculator.setAPY(10); // Set APY to 10%
const amountStaked = 1000; // Amount staked in tokens
const durationInDays = 30; // Duration in days

const rewards = calculator.calculateRewards(amountStaked, durationInDays);
const totalAmount = calculator.calculateTotalAmount(amountStaked, durationInDays);

console.log(`Rewards for staking ${amountStaked} tokens for ${durationInDays} days: ${rewards.toFixed(2)} tokens`);
console.log(`Total amount after staking: ${totalAmount.toFixed(2)} tokens`);
