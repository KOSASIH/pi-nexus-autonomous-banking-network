// controllers/GamificationController.js
const GamificationService = require('../services/GamificationService');
const User = require('../models/User');

const saveMoney = async (req, res) => {
  const user = req.user;
  const amount = req.body.amount;
  // Calculate points earned based on amount saved
  const points = Math.floor(amount / 100);
  await GamificationService.earnPoints(user, points);
  res.send({ message: 'Points earned!' });
};

const budgetSuccessfully = async (req, res) => {
  const user = req.user;
  // Unlock budgeting achievement badge
  await GamificationService.unlockBadge(user, 'budgetingAchievement');
  res.send({ message: 'Badge unlocked!' });
};

const makeLowCarbonTransaction = async (req, res) => {
  const user = req.user;
  // Calculate points earned based on low-carbon transaction
  const points = Math.floor(req.body.amount / 10);
  await GamificationService.earnPoints(user, points);
  res.send({ message: 'Points earned!' });
};

const redeemReward = async (req, res) => {
  const user = req.user;
  const rewardId = req.body.rewardId;
  const reward = await Reward.findById(rewardId);
  if (await GamificationService.redeemReward(user, reward)) {
    res.send({ message: 'Reward redeemed!' });
  } else {
    res.status(400).send({ message: 'Insufficient points' });
  }
};

module.exports = { saveMoney, budgetSuccessfully, makeLowCarbonTransaction, redeemReward };
