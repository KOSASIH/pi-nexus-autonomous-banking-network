// src/gamification/rewardsSystem.ts
export const calculateRewards = (transactionAmount: number) => {
    return Math.floor(transactionAmount * 0.01); // 1% reward
};
