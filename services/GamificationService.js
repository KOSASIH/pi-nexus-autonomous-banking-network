// services/GamificationService.js
const User = require("../models/User");
const Reward = require("../models/Reward");
const PartnerMerchant = require("../models/PartnerMerchant");

const earnPoints = async (user, points) => {
  user.gamification.points += points;
  await user.save();
};

const unlockBadge = async (user, badge) => {
  user.gamification.badges.push(badge);
  await user.save();
};

const redeemReward = async (user, reward) => {
  if (user.gamification.points >= reward.pointsRequired) {
    user.gamification.points -= reward.pointsRequired;
    user.gamification.rewards.push(reward);
    await user.save();
    return true;
  }
  return false;
};

module.exports = { earnPoints, unlockBadge, redeemReward };
