// rewards_system.js
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

const REWARDS_API_URL = 'https://api.rewards.com';

const getUserRewards = async (userId) => {
  try {
    const response = await axios.get(`${REWARDS_API_URL}/users/${userId}/rewards`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching user rewards: ${error.message}`);
    throw error;
  }
};

const addUserReward = async (userId, rewardType, amount) => {
  try {
    const rewardId = uuidv4();
    const response = await axios.post(`${REWARDS_API_URL}/rewards`, {
      userId,
      rewardType,
      amount,
      rewardId,
    });
    return response.data;
  } catch (error) {
    console.error(`Error adding user reward: ${error.message}`);
    throw error;
  }
};

module.exports = {
  getUserRewards,
  addUserReward,
};

// leaderboard.js
const axios = require('axios');

const LEADERBOARD_API_URL = 'https://api.leaderboard.com';

const getLeaderboard = async () => {
  try {
    const response = await axios.get(`${LEADERBOARD_API_URL}/leaderboard`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching leaderboard: ${error.message}`);
    throw error;
  }
};

module.exports = {
  getLeaderboard,
};

// challenges.js
const axios = require('axios');

const CHALLENGES_API_URL = 'https://api.challenges.com';

const getChallenges = async () => {
  try {
    const response = await axios.get(`${CHALLENGES_API_URL}/challenges`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching challenges: ${error.message}`);
    throw error;
  }
};

const submitChallengeResult = async (challengeId, userId, result) => {
  try {
    const response = await axios.post(`${CHALLENGES_API_URL}/results`, {
      challengeId,
      userId,
      result,
    });
    return response.data;
  } catch (error) {
    console.error(`Error submitting challenge result: ${error.message}`);
    throw error;
  }
};

module.exports = {
  getChallenges,
  submitChallengeResult,
};
